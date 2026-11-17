"""Lecture des CSV et chargement des enregistrements.

Chaque groupe a ses conventions (noms de dossiers hor / horizontal / vert / ver,
en-têtes aX / acc_x / X, ou pas d'en-tête). Ce module les unifie : à la sortie,
chaque enregistrement est un tableau (75, 6) dans l'ordre de config.CHANNELS.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CHANNELS, CLASS_KEYWORDS, CLASSES, COLUMN_ALIASES, DATA_DIR, N_SAMPLES


@dataclass
class Recordings:
    """Un ensemble d'enregistrements et leurs étiquettes."""
    X: np.ndarray        # (n, 75, 6) signaux bruts, non normalisés
    y: np.ndarray        # (n,) indice du geste dans CLASSES
    groups: np.ndarray   # (n,) nom du groupe, ex. "group3"
    files: list          # (n,) chemin de chaque CSV

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        idx = np.asarray(idx)
        return Recordings(self.X[idx], self.y[idx], self.groups[idx], [self.files[i] for i in idx])

    def group_names(self):
        return sorted(set(self.groups), key=_group_sort_key)


def _group_sort_key(name):
    m = re.search(r"\d+", name)
    return (int(m.group()) if m else 1_000_000, name)


def infer_class(path):
    """Geste déduit du nom du fichier, sinon du nom du dossier (casse ignorée)."""
    path = Path(path)
    for text in (path.stem.lower(), path.parent.name.lower()):
        for keyword, name in CLASS_KEYWORDS:
            if keyword in text:
                return name
    return None


def infer_group(path):
    """Groupe déduit d'un dossier parent nommé groupN, sinon du numéro au début du fichier."""
    path = Path(path)
    for parent in path.parents:
        if re.fullmatch(r"group\d+", parent.name.lower()):
            return parent.name.lower()
    m = re.match(r"(\d+)_", path.name)
    return f"group{m.group(1)}" if m else "?"


def read_csv(path):
    """Lit un CSV et renvoie un tableau (m, 6) dans l'ordre CHANNELS, ou None si format inconnu."""
    with open(path) as f:
        first = f.readline().strip()
    cells = [c.strip().strip('"').lower() for c in first.split(",")]
    try:  # pas d'en-tête : les colonnes sont déjà dans l'ordre
        [float(c) for c in cells]
        data = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
        return data if data.shape[1] == len(CHANNELS) else None
    except ValueError:
        pass
    order = []
    for aliases in COLUMN_ALIASES:
        idx = next((i for i, c in enumerate(cells) if c in aliases), None)
        if idx is None:
            return None
        order.append(idx)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return data[:, order]


def resample(signal, n=N_SAMPLES):
    """Ramène un enregistrement à exactement n mesures (interpolation linéaire)."""
    if len(signal) == n:
        return signal
    old_t = np.linspace(0.0, 1.0, len(signal))
    new_t = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(new_t, old_t, signal[:, k]) for k in range(signal.shape[1])]).astype(np.float32)


def load_recordings(root=DATA_DIR, verbose=True):
    """Charge tous les CSV sous `root` (récursivement).

    Les fichiers dont le nom apparaît deux fois ne sont gardés qu'une fois ;
    ceux dont le geste ou le format n'est pas reconnu sont ignorés.
    """
    root = Path(root)
    files = sorted(p for p in root.rglob("*.csv") if ".venv" not in p.parts)

    X, y, groups, kept = [], [], [], []
    seen, duplicates, ignored = set(), 0, []
    for path in files:
        key = path.name.lower()
        if key in seen:
            duplicates += 1
            continue
        name = infer_class(path)
        signal = read_csv(path)
        if name is None or signal is None or len(signal) < 2:
            ignored.append(path)
            continue
        seen.add(key)
        X.append(resample(signal))
        y.append(CLASSES.index(name))
        groups.append(infer_group(path))
        kept.append(str(path))

    if verbose:
        if duplicates:
            print(f"  {duplicates} doublon(s) ignoré(s) (même nom de fichier).")
        if ignored:
            print(f"  {len(ignored)} fichier(s) ignoré(s) (geste ou format non reconnu), par exemple :")
            for p in ignored[:5]:
                print("     ", p)
    if not X:
        raise SystemExit(f"Aucun fichier exploitable trouvé dans {root}")
    return Recordings(np.stack(X), np.array(y), np.array(groups), kept)


def describe(rec):
    """Résumé lisible : nombre d'enregistrements par groupe et par geste."""
    lines = [f"{len(rec)} enregistrements"]
    for g in rec.group_names():
        mask = rec.groups == g
        counts = ", ".join(f"{c} {(rec.y[mask] == k).sum()}" for k, c in enumerate(CLASSES))
        lines.append(f"  {g:8s} {mask.sum():5d}  ({counts})")
    return "\n".join(lines)
