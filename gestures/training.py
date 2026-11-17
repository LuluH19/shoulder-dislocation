"""Entraînement, prédiction, sauvegarde et chargement du modèle."""
import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import CHANNELS, CLASSES, MODEL_WIDTHS, N_SAMPLES
from .model import CNN1D
from .preprocessing import augment


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    augment: bool = True
    seed: int = 0
    widths: tuple = field(default=MODEL_WIDTHS)


def to_tensor(X):
    """(n, 75, axes) numpy -> (n, axes, 75) tensor, le format attendu par Conv1d."""
    return torch.as_tensor(np.asarray(X, dtype=np.float32)).permute(0, 2, 1).contiguous()


class GestureDataset(Dataset):
    """Rend un geste à la fois, avec augmentation aléatoire si demandée."""

    def __init__(self, X, y, augment=False):
        self.X = to_tensor(X)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        return (augment(x) if self.augment else x), self.y[i]


def _run_epoch(model, loader, loss_fn, optimizer=None):
    """Un passage sur les données. Avec optimizer : entraînement, sinon : évaluation."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            out = model(X)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def fit(X_train, y_train, X_val=None, y_val=None, config=None, verbose=False, init_model=None):
    """Entraîne un CNN1D sur des signaux déjà normalisés (n, 75, axes).

    Avec un jeu de validation, on garde l'époque dont la loss de validation est
    la plus basse ; sinon, la dernière. Avec init_model, on repart de ses poids
    (une copie : init_model n'est pas modifié) au lieu de poids aléatoires.
    Renvoie (modèle en mode évaluation, historique).
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    if init_model is not None:
        model = copy.deepcopy(init_model)
    else:
        model = CNN1D(n_channels=np.asarray(X_train).shape[2], widths=config.widths)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_loader = DataLoader(GestureDataset(X_train, y_train, augment=config.augment),
                              batch_size=config.batch_size, shuffle=True)
    has_val = X_val is not None and len(X_val) > 0
    val_loader = DataLoader(GestureDataset(X_val, y_val), batch_size=256) if has_val else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state = float("inf"), None
    for epoch in range(1, config.epochs + 1):
        tl, ta = _run_epoch(model, train_loader, loss_fn, optimizer)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        if has_val:
            vl, va = _run_epoch(model, val_loader, loss_fn)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            if vl < best_loss:
                best_loss, best_state = vl, copy.deepcopy(model.state_dict())
        if verbose and (epoch == 1 or epoch % 10 == 0):
            msg = f"  époque {epoch:3d}/{config.epochs}  train loss {tl:.3f} acc {ta * 100:5.1f} %"
            if has_val:
                msg += f" | val loss {vl:.3f} acc {va * 100:5.1f} %"
            print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), history


@torch.no_grad()
def predict(model, X):
    """Signaux normalisés (n, 75, axes) -> (classes prédites (n,), probabilités (n, 3))."""
    model.eval()
    probs = torch.softmax(model(to_tensor(X)), dim=1).numpy()
    return probs.argmax(1), probs


def save_model(path, model, config, **meta):
    """Sauvegarde les poids et tout ce qu'il faut pour les réutiliser (gestes, axes, normalisation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        "channels": CHANNELS,
        "n_samples": N_SAMPLES,
        "norm": "record",  # z-score par enregistrement et par axe (preprocessing.normalize)
        "widths": list(model.widths),
        "config": {**asdict(config), "widths": list(config.widths)},
        **meta,
    }, path)


def load_model(path):
    """Renvoie (modèle en mode évaluation, checkpoint complet)."""
    ckpt = torch.load(path, map_location="cpu")
    widths = ckpt.get("widths") or _widths_from_state(ckpt["state_dict"])
    model = CNN1D(n_channels=len(ckpt.get("channels", CHANNELS)), n_classes=len(ckpt["classes"]), widths=widths)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def _widths_from_state(state_dict):
    """Pour les anciens fichiers .pt qui n'enregistraient pas les largeurs."""
    return tuple(state_dict[f"features.{i}.weight"].shape[0] for i in (0, 4, 8))


def add_training_args(parser, epochs=60):
    """Options communes à tous les scripts qui entraînent un modèle."""
    g = parser.add_argument_group("entraînement")
    g.add_argument("--epochs", type=int, default=epochs)
    g.add_argument("--batch", type=int, default=32, help="taille d'un lot")
    g.add_argument("--lr", type=float, default=1e-3, help="taux d'apprentissage")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-augment", action="store_true", help="désactive l'augmentation")
    g.add_argument("--widths", type=int, nargs=3, default=list(MODEL_WIDTHS),
                   help="filtres des 3 convolutions (la carte attend 16 32 64)")


def config_from_args(args):
    return TrainConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr, seed=args.seed,
                       augment=not args.no_augment, widths=tuple(args.widths))
