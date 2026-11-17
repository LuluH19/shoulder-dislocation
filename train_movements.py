"""Entraînement d'un CNN 1D (PyTorch) pour reconnaître les mouvements circle / horizontal / vertical.

Entrée : des fichiers CSV de 6 colonnes (acc X/Y/Z puis gyro X/Y/Z), ~75 lignes chacun,
         rangés dans des dossiers (circle / hor / horizontal / ver / vert / vertical ...).
         Les en-têtes reconnus : aX..gZ (group1, group2), acc_x..gyro_z (group3),
         X,Y,Z,GX,GY,GZ (group5), ou pas d'en-tête du tout.
         La classe est lue dans le nom du fichier, sinon dans le nom du dossier.
         Le numéro de groupe est le nombre au début du nom du fichier.

Utilisation (depuis la racine du projet) :
    .venv/bin/python train_movements.py --data .                       # tous les groupes
    .venv/bin/python train_movements.py --data . --holdout-group 3     # test = tous les fichiers du groupe 3
    .venv/bin/python train_movements.py --data dataset_group2 --test-data dataset_group3
        # entraîne/valide sur group2, puis teste EN PLUS sur group3 (jamais vu)
    .venv/bin/python train_movements.py --data . --epochs 80 --lr 0.001
    .venv/bin/python train_movements.py --data . --norm global         # normalisation d'origine du script

Normalisation (--norm) :
    record (défaut) : z-score par enregistrement et par axe, (x - moyenne) / écart-type de la fenêtre.
                      Indispensable ici car group3 (z-score) et group5 (min-max) sont déjà
                      normalisés différemment : c'est la seule façon de rendre les groupes comparables.
    global          : x - moyenne de la fenêtre, puis division par un écart-type global du train
                      (garde l'amplitude, mais seulement valable si tous les groupes ont les mêmes unités).

Sorties (dossier models/ par défaut) :
    movements.pt            modèle entraîné (+ classes + normalisation)
    scaler.json             normalisation à reproduire sur la carte
    movements.onnx          modèle exporté (si onnx + onnxscript sont installés)
    learning_curves.png     courbes de loss / accuracy
    confusion_matrix.png    matrice de confusion sur le jeu de test
    confusion_matrix_external.png   idem sur le dataset externe (--test-data)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # évite un blocage torch sur macOS

import argparse
import glob
import json
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
# noms de colonnes utilisés par les différents groupes (en minuscules) -> index du canal
COLUMN_ALIASES = {
    0: ["ax", "acc_x", "x", "accx"],
    1: ["ay", "acc_y", "y", "accy"],
    2: ["az", "acc_z", "z", "accz"],
    3: ["gx", "gyro_x", "gyrox"],
    4: ["gy", "gyro_y", "gyroy"],
    5: ["gz", "gyro_z", "gyroz"],
}
# même ordre que le notebook Jupyter/test.ipynb et compare_group3.py
CLASSES = ["circle", "horizontal", "vertical"]
# mot-clé (dans le nom du fichier ou du dossier) -> classe. "ver" couvre ver / vert / vertical.
CLASS_KEYS = (("circ", "circle"), ("hor", "horizontal"), ("ver", "vertical"))
N_SAMPLES = 75  # 1,5 s à 50 Hz


# ==============================================================================
# 1. CHARGEMENT DES DONNÉES
# ==============================================================================
def infer_class(path):
    """Classe déduite du nom de fichier d'abord, puis du nom du dossier (casse ignorée)."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    folder = os.path.basename(os.path.dirname(path)).lower()
    for text in (stem, folder):
        for key, name in CLASS_KEYS:
            if key in text:
                return name
    return None


def infer_group(path):
    m = re.match(r"(\d+)", os.path.basename(path))
    return m.group(1) if m else "?"


def read_signal(path):
    """Lit un CSV et renvoie un tableau (m, 6) dans l'ordre CHANNELS, ou None si format inconnu."""
    with open(path) as f:
        first = f.readline().strip()
    cells = [c.strip().strip('"').lower() for c in first.split(",")]
    try:  # fichier sans en-tête ?
        [float(c) for c in cells]
        data = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
        return data if data.shape[1] == len(CHANNELS) else None
    except ValueError:
        pass
    order = []
    for k in range(len(CHANNELS)):
        idx = next((i for i, c in enumerate(cells) if c in COLUMN_ALIASES[k]), None)
        if idx is None:
            return None
        order.append(idx)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return data[:, order]


def resample(sig, n=N_SAMPLES):
    """Ramène un enregistrement à exactement n mesures (interpolation linéaire)."""
    if len(sig) == n:
        return sig
    old_t = np.linspace(0.0, 1.0, len(sig))
    new_t = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        [np.interp(new_t, old_t, sig[:, k]) for k in range(sig.shape[1])]
    ).astype(np.float32)


def load_all(root):
    files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    files = [p for p in files if os.sep + ".venv" + os.sep not in p]
    # les dossiers "processed" passent en dernier : en cas de doublon, on garde le fichier brut
    files.sort(key=lambda p: ("processed" in p.lower(), p))

    X, y, groups, names = [], [], [], []
    seen, doublons, ignores = set(), 0, []
    for p in files:
        key = os.path.basename(p).lower()
        if key in seen:
            doublons += 1
            continue
        cls = infer_class(p)
        sig = read_signal(p)
        if cls is None or sig is None or len(sig) < 2:
            ignores.append(p)
            continue
        seen.add(key)
        X.append(resample(sig))
        y.append(CLASSES.index(cls))
        groups.append(infer_group(p))
        names.append(os.path.basename(p))

    if doublons:
        print(f"  {doublons} doublon(s) ignoré(s) (même nom de fichier).")
    if ignores:
        print(f"  {len(ignores)} fichier(s) ignoré(s) (classe ou format non reconnu), par exemple :")
        for p in ignores[:5]:
            print("     ", p)
    if not X:
        raise SystemExit(f"Aucun fichier exploitable trouvé dans {root}")
    return np.stack(X), np.array(y), np.array(groups), names


def normalize(X, mode, std=None):
    """record : z-score par enregistrement et par axe. global : centrage puis / std (fourni)."""
    X = X - X.mean(axis=1, keepdims=True)
    if mode == "record":
        return X / (X.std(axis=1, keepdims=True) + 1e-8)
    return X / std


def split(y, groups, holdout, seed):
    """Découpe par fichier, équilibrée par classe : 70 % train / 15 % val / 15 % test.
    Avec --holdout-group, le test = tous les fichiers de ce groupe (personne jamais vue)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    test = list(idx[groups == holdout]) if holdout else []
    rest = idx[groups != holdout] if holdout else idx
    train, val = [], []
    for c in range(len(CLASSES)):
        ids = rest[y[rest] == c]
        rng.shuffle(ids)
        n_val = max(1, round(0.15 * len(ids)))
        n_test = 0 if holdout else max(1, round(0.15 * len(ids)))
        val += list(ids[:n_val])
        test += [] if holdout else list(ids[n_val:n_val + n_test])
        train += list(ids[n_val + n_test:])
    return np.array(train), np.array(val), np.array(test)


# ==============================================================================
# 2. DATASET (avec augmentation pour l'entraînement uniquement)
# ==============================================================================
def shift(x, s):
    """Décale le signal de s mesures dans le temps, en répétant le bord (pas de bouclage)."""
    if s > 0:
        return torch.cat([x[:, :1].expand(-1, s), x[:, :-s]], dim=1)
    if s < 0:
        return torch.cat([x[:, -s:], x[:, -1:].expand(-1, -s)], dim=1)
    return x


class Mouvements(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).contiguous()  # (n, 6, 75)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = x * (0.85 + 0.3 * torch.rand(1))                  # amplitude ±15 %
            x = shift(x, int(torch.randint(-6, 7, (1,))))         # décalage ±0,12 s
            x = x + 0.05 * torch.randn_like(x)                    # bruit
        return x, self.y[i]


# ==============================================================================
# 3. MODÈLE : petit CNN 1D (~10 000 paramètres, assez léger pour une carte ESP32)
# ==============================================================================
class CNN1D(nn.Module):
    def __init__(self, n_classes, in_channels=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):  # x : (batch, 6, 75)
        return self.classifier(self.features(x))


# ==============================================================================
# 4. ENTRAÎNEMENT ET ÉVALUATION
# ==============================================================================
def train_epoch(loader, model, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (pred.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(loader, model, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += loss_fn(pred, y).item() * len(y)
        correct += (pred.argmax(1) == y).sum().item()
        total += len(y)
        preds += pred.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()
    return total_loss / total, correct / total, np.array(labels), np.array(preds)


def confusion(labels, preds, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def save_plots(history, cm, out_dir, cm_ext=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : courbes non sauvegardées (pip install matplotlib).")
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.title("Loss"); plt.xlabel("Époque"); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation")
    plt.title("Accuracy (%)"); plt.xlabel("Époque"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"))
    plt.close()

    for matrix, fname, title in ((cm, "confusion_matrix.png", "Matrice de confusion (test interne)"),
                                 (cm_ext, "confusion_matrix_external.png", "Matrice de confusion (test EXTERNE)")):
        if matrix is None:
            continue
        plt.figure(figsize=(5, 4.5))
        plt.imshow(matrix, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, matrix[i, j], ha="center", va="center",
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        plt.xticks(range(len(CLASSES)), CLASSES)
        plt.yticks(range(len(CLASSES)), CLASSES)
        plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


def report(title, cm):
    """Précision globale, précision et recall par classe, matrice de confusion."""
    tp = np.diag(cm).astype(float)
    acc = tp.sum() / max(cm.sum(), 1)
    print(f"\n{title} : précision (accuracy) {acc * 100:.1f}%")
    print("Matrice de confusion (lignes = réel, colonnes = prédit) :", CLASSES)
    print(cm)
    for c, name in enumerate(CLASSES):
        prec = tp[c] / cm[:, c].sum() if cm[:, c].sum() else float("nan")
        rec = tp[c] / cm[c].sum() if cm[c].sum() else float("nan")
        print(f"  {name:10s}: précision {prec * 100:5.1f}%  recall {rec * 100:5.1f}%  ({int(tp[c])}/{cm[c].sum()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="dossier contenant les CSV (parcouru récursivement)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", choices=["record", "global"], default="record",
                    help="record = z-score par enregistrement (défaut), global = écart-type global du train")
    ap.add_argument("--holdout-group", default=None, help="numéro de groupe réservé au test")
    ap.add_argument("--test-data", default=None,
                    help="dossier d'un AUTRE dataset (autre groupe) utilisé uniquement pour le test, jamais pour l'entraînement/validation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Matériel utilisé : {device}")

    # ---- données ----
    print(f"Chargement de {args.data} ...")
    X, y, groups, names = load_all(args.data)
    print(f"{len(y)} enregistrements, groupes : {sorted(set(map(str, groups)))}")
    for c, name in enumerate(CLASSES):
        print(f"  {name:10s}: {(y == c).sum()}")

    if args.holdout_group is not None and args.holdout_group not in set(groups):
        raise SystemExit(f"Groupe {args.holdout_group} absent des données ({sorted(set(groups))}).")
    train_idx, val_idx, test_idx = split(y, groups, args.holdout_group, args.seed)
    if len(test_idx) == 0:
        raise SystemExit("Jeu de test vide : vérifie --holdout-group.")
    print(f"Train : {len(train_idx)} | Validation : {len(val_idx)} | Test : {len(test_idx)}")

    std = np.ones(len(CHANNELS), dtype=np.float32)
    if args.norm == "global":
        # écart-type calculé sur le TRAIN uniquement (puis appliqué partout, y compris sur la carte)
        Xc = X - X.mean(axis=1, keepdims=True)
        std = Xc[train_idx].reshape(-1, len(CHANNELS)).std(axis=0)
        std[std < 1e-9] = 1.0
        if len(set(groups)) > 1:
            print("Remarque : --norm global mélange des groupes aux unités différentes "
                  "(group3 et group5 sont déjà normalisés). --norm record est conseillé.")
    X = normalize(X, args.norm, std)

    X_ext = y_ext = None
    if args.test_data:
        print(f"Chargement du dataset de test externe {args.test_data} ...")
        X_ext, y_ext, g_ext, _ = load_all(args.test_data)
        print(f"  {len(y_ext)} enregistrements, groupes : {sorted(set(map(str, g_ext)))}")
        common = set(map(str, groups)) & set(map(str, g_ext))
        if common:
            print(f"  ATTENTION : groupes en commun avec le train {sorted(common)} : le test externe n'est pas indépendant.")
        X_ext = normalize(X_ext, args.norm, std)

    train_loader = DataLoader(Mouvements(X[train_idx], y[train_idx], augment=True), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(Mouvements(X[val_idx], y[val_idx]), batch_size=args.batch)
    test_loader = DataLoader(Mouvements(X[test_idx], y[test_idx]), batch_size=args.batch)

    # ---- modèle ----
    model = CNN1D(len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : CNN 1D, {n_params} paramètres")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # ---- entraînement ----
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state = float("inf"), None
    for t in range(args.epochs):
        tl, ta = train_epoch(train_loader, model, loss_fn, optimizer, device)
        vl, va, _, _ = evaluate(val_loader, model, loss_fn, device)
        for k, v in zip(history, (tl, vl, ta, va)):
            history[k].append(v)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (t + 1) % 5 == 0 or t == 0:
            print(f"Époque {t + 1:3d}/{args.epochs} -> train loss {tl:.3f} acc {ta * 100:.1f}% | "
                  f"val loss {vl:.3f} acc {va * 100:.1f}%")
    model.load_state_dict(best_state)  # on garde la meilleure époque (loss de validation minimale)

    # ---- évaluation finale ----
    _, _, labels, preds = evaluate(test_loader, model, loss_fn, device)
    cm = confusion(labels, preds, len(CLASSES))
    title = (f"Jeu de TEST : groupe {args.holdout_group} (jamais vu)" if args.holdout_group
             else "Jeu de TEST interne (même dataset que l'entraînement)")
    report(title, cm)
    cm_ext = None
    if X_ext is not None:
        ext_loader = DataLoader(Mouvements(X_ext, y_ext), batch_size=args.batch)
        _, _, labels_e, preds_e = evaluate(ext_loader, model, loss_fn, device)
        cm_ext = confusion(labels_e, preds_e, len(CLASSES))
        report("Jeu de TEST EXTERNE (autre dataset, jamais vu)", cm_ext)
    save_plots(history, cm, args.out, cm_ext)

    # ---- sauvegardes ----
    model.to("cpu").eval()
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "norm": args.norm,
                "std": std.tolist()}, os.path.join(args.out, "movements.pt"))
    usage = ("x_norm = (x - moyenne_de_la_fenêtre) / écart-type_de_la_fenêtre (par axe)"
             if args.norm == "record" else "x_norm = (x - moyenne_de_la_fenêtre) / std")
    with open(os.path.join(args.out, "scaler.json"), "w") as f:
        json.dump({"channels": CHANNELS, "n_samples": N_SAMPLES, "norm": args.norm,
                   "std": [float(s) for s in std], "classes": CLASSES,
                   "usage": usage + " ; entrée du modèle : (1, 6, 75)"},
                  f, indent=2, ensure_ascii=False)
    try:
        onnx_path = os.path.join(args.out, "movements.onnx")
        torch.onnx.export(model, torch.randn(1, len(CHANNELS), N_SAMPLES), onnx_path,
                          input_names=["input"], output_names=["logits"], opset_version=18)
        import onnx
        onnx.save(onnx.load(onnx_path), onnx_path)  # regroupe les poids dans un seul fichier .onnx
        if os.path.exists(onnx_path + ".data"):
            os.remove(onnx_path + ".data")
        print(f"Export ONNX : {onnx_path}")
    except Exception as e:  # onnx / onnxscript absents, par exemple
        print(f"Export ONNX non fait ({type(e).__name__}). Essaie : pip install onnx onnxscript")
    print(f"Fichiers sauvegardés dans {args.out}/")


if __name__ == "__main__":
    main()
