"""Compare le dataset group3 a tous les autres groupes et sauvegarde les PNG dans screenshot/.

Usage :
    .venv/bin/python compare_group3.py              # 40 epochs par modele
    .venv/bin/python compare_group3.py --epochs 20

Les groupes sont detectes automatiquement (tout dossier dataset_group*), donc un
nouveau groupe ajoute au projet est compare sans toucher au script.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"   # evite un blocage torch + scikit-learn sur macOS

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # pas de fenetre : on ecrit seulement des fichiers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "screenshot"
REF = "group3"
CLASSES = ["circle", "horizontal", "vertical"]
# noms de dossiers utilises par les differents groupes pour chaque classe
ALIAS = {
    "circle": ["circle"],
    "horizontal": ["horizontal", "hor"],
    "vertical": ["vertical", "vert", "ver"],
}
AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
COULEUR_REF = "#eb6834"
COULEURS_AUTRES = ["#2a78d6", "#1f9e89", "#8e5bd0", "#c4a000", "#5f6b7a"]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "savefig.dpi": 150,
                     "savefig.bbox": "tight"})


# ---------------------------------------------------------------- donnees
def normaliser(signal):
    # z-score par axe et par enregistrement : rend comparables les groupes
    # qui n'ont pas la meme unite / echelle (ex. group5 deja en min-max)
    return (signal - signal.mean(0)) / (signal.std(0) + 1e-8)


def dossier_classe(dossier_groupe, classe):
    for nom in ALIAS[classe]:
        if (dossier_groupe / nom).is_dir():
            return dossier_groupe / nom
    return None


def charger(dossier_groupe):
    X, y = [], []
    for label, classe in enumerate(CLASSES):
        d = dossier_classe(dossier_groupe, classe)
        if d is None:
            raise FileNotFoundError(f"{dossier_groupe.name} : pas de dossier pour '{classe}'")
        for f in sorted(glob.glob(str(d / "*.csv"))):
            X.append(normaliser(np.loadtxt(f, delimiter=",", skiprows=1)))
            y.append(label)
    return np.stack(X).astype(np.float32), np.array(y)


def trouver_groupes():
    groupes = {}
    for d in sorted(ROOT.glob("dataset_group*")):
        if d.is_dir():
            groupes[d.name.replace("dataset_", "")] = d
    if REF not in groupes:
        raise SystemExit(f"dataset_{REF} introuvable dans {ROOT}")
    # group3 en premier, puis les autres dans l'ordre
    return {REF: groupes.pop(REF), **groupes}


# ---------------------------------------------------------------- modele
class CNN(nn.Module):
    def __init__(self, n_axes=6, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_axes, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),     nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),     nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def entrainer_cnn(X_np, y_np, epochs):
    torch.manual_seed(42)
    Xt = torch.tensor(X_np).permute(0, 2, 1)
    m = CNN(n_axes=Xt.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, torch.tensor(y_np)), batch_size=32, shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(m(xb), yb).backward()
            opt.step()
    return m.eval()


def predire(m, X_np):
    with torch.no_grad():
        return m(torch.tensor(X_np).permute(0, 2, 1)).argmax(1).numpy()


def couleur(g, groupes):
    if g == REF:
        return COULEUR_REF
    autres = [x for x in groupes if x != REF]
    return COULEURS_AUTRES[autres.index(g) % len(COULEURS_AUTRES)]


def sauver(fig, nom):
    chemin = OUT / nom
    fig.savefig(chemin)
    plt.close(fig)
    print("  ->", chemin.relative_to(ROOT))


# ---------------------------------------------------------------- figures
def fig_accuracy_ref(resultats, groupes):
    """Modele group3 teste sur son test et sur chaque autre groupe."""
    noms = list(resultats[REF])
    acc = [accuracy_score(*resultats[REF][g]) * 100 for g in noms]
    f1 = [f1_score(*resultats[REF][g], average="macro", zero_division=0) * 100 for g in noms]
    labels = [f"{g}\n({'test' if g == REF else 'jamais vu'})" for g in noms]

    fig, ax = plt.subplots(figsize=(1.8 * len(noms) + 2, 4.2))
    barres = ax.bar(range(len(noms)), acc, 0.6, color=[couleur(g, groupes) for g in noms])
    ax.bar_label(barres, labels=[f"{a:.0f} %\nF1 {f:.0f}" for a, f in zip(acc, f1)],
                 fontsize=9, color="#3d3d3a")
    ax.axhline(100 / len(CLASSES), color="#9a9a94", lw=1, ls="--")
    ax.text(len(noms) - 0.5, 100 / len(CLASSES) + 1.5, "hasard", ha="right",
            fontsize=8, color="#9a9a94")
    ax.set_xticks(range(len(noms)), labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Modèle {REF} testé sur chaque groupe")
    sauver(fig, "01_accuracy_modele_group3.png")


def fig_confusions_ref(resultats):
    noms = list(resultats[REF])
    fig, axs = plt.subplots(1, len(noms), figsize=(4.2 * len(noms), 4.2))
    for ax, g in zip(np.atleast_1d(axs), noms):
        ye, p = resultats[REF][g]
        ConfusionMatrixDisplay(confusion_matrix(ye, p, labels=range(len(CLASSES))),
                               display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.grid(False)
        ax.set_title(f"Modèle {REF} -> {g}\naccuracy = {accuracy_score(ye, p):.3f}", fontsize=10)
    fig.tight_layout()
    sauver(fig, "02_confusion_modele_group3.png")


def fig_matrice_croisee(resultats, groupes):
    """Ligne = groupe d'entrainement, colonne = groupe de test. Diagonale = test held-out."""
    noms = list(groupes)
    M = np.array([[accuracy_score(*resultats[tr][te]) * 100 for te in noms] for tr in noms])
    fig, ax = plt.subplots(figsize=(1.3 * len(noms) + 2.5, 1.1 * len(noms) + 1.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(noms)):
        for j in range(len(noms)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > 60 else "#1f1f1d",
                    fontweight="bold" if REF in (noms[i], noms[j]) else "normal")
    ax.set_xticks(range(len(noms)), noms)
    ax.set_yticks(range(len(noms)), noms)
    ax.set_xlabel("testé sur")
    ax.set_ylabel("entraîné sur")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    i = noms.index(REF)
    ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(noms), 1, fill=False, ec=COULEUR_REF, lw=2))
    ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, len(noms), fill=False, ec=COULEUR_REF, lw=2))
    fig.colorbar(im, ax=ax, label="accuracy (%)", shrink=0.8)
    ax.set_title("Accuracy croisée entre groupes (diagonale = test interne)")
    sauver(fig, "03_matrice_croisee.png")


def fig_signal_moyen(data, groupes):
    """Signal moyen (+/- 1 ecart-type) par axe, group3 vs chaque autre groupe."""
    t = np.arange(next(iter(data.values()))[0].shape[1])
    for k, classe in enumerate(CLASSES):
        fig, axs = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
        for a, ax in enumerate(axs.ravel()):
            for g, (X, y) in data.items():
                S = X[y == k][:, :, a]
                mu, sd = S.mean(0), S.std(0)
                c = couleur(g, groupes)
                ax.plot(t, mu, color=c, lw=2.4 if g == REF else 1.4, label=g,
                        zorder=3 if g == REF else 2)
                if g == REF:
                    ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
            ax.set_title(AXES[a], fontsize=10)
        for ax in axs[1]:
            ax.set_xlabel("échantillon")
        axs[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{classe} : signal moyen normalisé ({REF} avec ±1 écart-type)")
        fig.tight_layout()
        sauver(fig, f"04_signal_moyen_{classe}.png")


def fig_vitesse_circle(data, groupes):
    """Nombre de tours en 1,5 s via le pic FFT de acc_x + acc_y."""
    k = CLASSES.index("circle")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    largeur = 0.8 / len(data)
    for i, (g, (X, y)) in enumerate(data.items()):
        A = X[y == k]
        puissance = (np.abs(np.fft.rfft(A[:, :, :2], axis=1)) ** 2).sum(2)
        n = puissance[:, 1:8].argmax(1) + 1
        pct = [np.mean(n == t) * 100 for t in (1, 2, 3)] + [np.mean(n >= 4) * 100]
        barres = ax.bar(np.arange(4) + (i - (len(data) - 1) / 2) * largeur, pct, largeur * 0.95,
                        color=couleur(g, groupes), label=g)
        ax.bar_label(barres, fmt="%.0f", fontsize=7, color="#3d3d3a")
    ax.set_xticks(range(4), ["1 tour\n(lent)", "2 tours", "3 tours", "4+\n(rapide)"])
    ax.set_ylabel("% des enregistrements circle")
    ax.set_title("Vitesse : nombre de cercles faits en 1,5 s")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sauver(fig, "05_vitesse_circle.png")


def fig_correlation_axes(data, groupes):
    """Correlation moyenne entre les 6 axes, par mouvement et par groupe.
    Insensible a la normalisation : montre si les axes bougent ensemble de la meme facon."""
    noms = list(groupes)
    fig, axs = plt.subplots(len(CLASSES), len(noms), figsize=(3.3 * len(noms), 3.3 * len(CLASSES)))
    courts = ["ax", "ay", "az", "gx", "gy", "gz"]
    for k, classe in enumerate(CLASSES):
        ref = None
        for j, g in enumerate(noms):
            X, y = data[g]
            S = X[y == k]
            C = np.einsum("nta,ntb->ab", S, S) / (S.shape[0] * S.shape[1])  # deja z-score
            if g == REF:
                ref = C
            ax = axs[k, j]
            ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
            for a in range(6):
                for b in range(6):
                    ax.text(b, a, f"{C[a, b]:.1f}".replace("0.", ".").replace("-.0", ".0"),
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(C[a, b]) > 0.6 else "#1f1f1d")
            ax.set_xticks(range(6), courts, fontsize=8)
            ax.set_yticks(range(6), courts, fontsize=8)
            ax.grid(False)
            titre = f"{g} - {classe}"
            if g != REF:
                titre += f"\nécart vs {REF} = {np.abs(C - ref).mean():.2f}"
            ax.set_title(titre, fontsize=9, color=COULEUR_REF if g == REF else "#1f1f1d")
    fig.suptitle("Corrélation entre axes (rouge = bougent ensemble, bleu = en opposition)")
    fig.tight_layout()
    sauver(fig, "06_correlation_axes.png")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    groupes = trouver_groupes()
    print("Groupes :", ", ".join(groupes))

    data, splits = {}, {}
    for g, d in groupes.items():
        X, y = charger(d)
        data[g] = (X, y)
        splits[g] = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        print(f"  {g}: {X.shape[0]} enregistrements")

    # resultats[train][test] = (y_vrai, y_pred)
    # meme groupe -> on teste sur le split test ; autre groupe -> sur tout le groupe
    resultats = {}
    for tr in groupes:
        Xtr, _, ytr, _ = splits[tr]
        print(f"Entraînement modèle {tr} ({len(Xtr)} enreg., {args.epochs} epochs)...")
        m = entrainer_cnn(Xtr, ytr, args.epochs)
        resultats[tr] = {}
        for te in groupes:
            if te == tr:
                _, Xte, _, yte = splits[te]
            else:
                Xte, yte = data[te]
            resultats[tr][te] = (yte, predire(m, Xte))

    print(f"\n{'entraîné':10s} {'testé sur':10s} {'n':>4s} {'acc':>6s} {'f1':>6s}")
    for tr in groupes:
        for te in groupes:
            if REF not in (tr, te):
                continue
            ye, p = resultats[tr][te]
            print(f"{tr:10s} {te:10s} {len(ye):4d} {accuracy_score(ye, p):6.3f} "
                  f"{f1_score(ye, p, average='macro', zero_division=0):6.3f}")

    print(f"\nSauvegarde dans {OUT.relative_to(ROOT)}/ :")
    fig_accuracy_ref(resultats, groupes)
    fig_confusions_ref(resultats)
    fig_matrice_croisee(resultats, groupes)
    fig_signal_moyen(data, groupes)
    fig_vitesse_circle(data, groupes)
    fig_correlation_axes(data, groupes)


if __name__ == "__main__":
    main()
