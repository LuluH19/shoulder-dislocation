"""Préparation des données : normalisation, découpage, augmentation."""
import numpy as np
import torch


def normalize(X):
    """Z-score par enregistrement et par axe : moyenne 0, écart-type 1.

    Ne garde que la forme du geste (pas sa force ni son unité), ce qui rend
    comparables les groupes déjà normalisés différemment (group3 en z-score,
    group5 en min-max). Accepte (75, 6) ou (n, 75, 6).
    Le firmware refait exactement ce calcul (normalize_window dans cnn.h).
    """
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=-2, keepdims=True)
    std = X.std(axis=-2, keepdims=True)
    return ((X - mean) / (std + 1e-8)).astype(np.float32)


def split(y, groups=None, val_frac=0.15, test_frac=0.15, holdout_group=None, seed=0):
    """Découpe les indices en entraînement / validation / test, équilibrés par geste.

    Avec holdout_group, le test contient tous les enregistrements de ce groupe
    (une personne jamais vue) et le reste est découpé en entraînement / validation.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    if holdout_group is not None:
        test = list(idx[groups == holdout_group])
        rest = idx[groups != holdout_group]
        test_frac = 0.0
    else:
        test, rest = [], idx
    train, val = [], []
    for c in np.unique(y):
        ids = rest[y[rest] == c]
        rng.shuffle(ids)
        n_val = round(val_frac * len(ids)) if val_frac else 0
        n_test = round(test_frac * len(ids)) if test_frac else 0
        val += list(ids[:n_val])
        test += list(ids[n_val:n_val + n_test])
        train += list(ids[n_val + n_test:])
    return np.array(train), np.array(val, dtype=int), np.array(test, dtype=int)


def shift(x, s):
    """Décale le signal (canaux, temps) de s mesures, en répétant le bord."""
    if s > 0:
        return torch.cat([x[:, :1].expand(-1, s), x[:, :-s]], dim=1)
    if s < 0:
        return torch.cat([x[:, -s:], x[:, -1:].expand(-1, -s)], dim=1)
    return x


def augment(x):
    """Variante aléatoire d'un geste (canaux, temps), utilisée seulement à l'entraînement."""
    x = x * (0.85 + 0.3 * torch.rand(1))                 # amplitude ±15 %
    x = shift(x, int(torch.randint(-6, 7, (1,))))        # décalage ±6 mesures (±0,12 s)
    return x + 0.05 * torch.randn_like(x)                # petit bruit
