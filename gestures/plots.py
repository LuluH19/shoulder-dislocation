"""Toutes les figures du projet. Chaque fonction écrit un PNG et renvoie son chemin."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on écrit seulement des fichiers
import matplotlib.pyplot as plt
import numpy as np

from . import analysis
from .config import CHANNELS, CLASSES

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})

REF_COLOR = "#eb6834"
OTHER_COLORS = ["#2a78d6", "#1f9e89", "#8e5bd0", "#c4a000", "#5f6b7a"]
TEXT_COLOR = "#3d3d3a"


def group_colors(names, ref=None):
    """Une couleur par groupe ; le groupe de référence est toujours orange."""
    others = [g for g in names if g != ref]
    return {g: REF_COLOR if g == ref else OTHER_COLORS[others.index(g) % len(OTHER_COLORS)] for g in names}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _heatmap_cells(ax, M, fmt, dark_above, fontsize=11, bold=None):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=fontsize,
                    color="white" if M[i, j] > dark_above else "#1f1f1d",
                    fontweight="bold" if bold and bold(i, j) else "normal")


# ---------------------------------------------------------------- entraînement
def learning_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(epochs, history["train_loss"], label="entraînement")
    a2.plot(epochs, [a * 100 for a in history["train_acc"]], label="entraînement")
    if history.get("val_loss"):
        a1.plot(epochs, history["val_loss"], label="validation")
        a2.plot(epochs, [a * 100 for a in history["val_acc"]], label="validation")
    a1.set_title("Loss")
    a2.set_title("Accuracy (%)")
    for ax in (a1, a2):
        ax.set_xlabel("époque")
        ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def confusion(cm, title, path, ax=None):
    """Matrice de confusion. Avec ax, dessine dans un sous-graphique sans sauvegarder."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    _heatmap_cells(ax, cm, lambda v: f"{v:d}", cm.max() / 2)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai geste")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    if own:
        fig.tight_layout()
        return _save(fig, path)


def confusion_grid(items, path, suptitle=None):
    """Plusieurs matrices côte à côte. items = [(titre, cm), ...]."""
    fig, axs = plt.subplots(1, len(items), figsize=(4.4 * len(items), 4.3))
    for ax, (title, cm) in zip(np.atleast_1d(axs), items):
        confusion(cm, title, None, ax=ax)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- généralisation
def accuracy_bars(names, values, path, title, colors=None, labels=None, xlabels=None):
    """Barres d'accuracy (%) avec la ligne du hasard (33 %)."""
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 2, 4.2))
    bars = ax.bar(range(len(names)), values, 0.6, color=colors or OTHER_COLORS[0])
    ax.bar_label(bars, labels=labels or [f"{v:.0f} %" for v in values], fontsize=9, color=TEXT_COLOR)
    chance = 100 / len(CLASSES)
    ax.axhline(chance, color="#9a9a94", lw=1, ls="--")
    ax.text(len(names) - 0.5, chance + 1.5, "hasard", ha="right", fontsize=8, color="#9a9a94")
    ax.set_xticks(range(len(names)), xlabels or names)
    ax.set_ylim(0, 115)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    return _save(fig, path)


def cross_accuracy(M, names, path, ref=None):
    """M[i, j] = accuracy (%) du modèle entraîné sur names[i], testé sur names[j]."""
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2.5, 1.1 * len(names) + 1.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    _heatmap_cells(ax, M, lambda v: f"{v:.0f}", 60,
                   bold=(lambda i, j: ref in (names[i], names[j])) if ref else None)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("testé sur")
    ax.set_ylabel("entraîné sur")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if ref in names:
        i = names.index(ref)
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(names), 1, fill=False, ec=REF_COLOR, lw=2))
        ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, len(names), fill=False, ec=REF_COLOR, lw=2))
    fig.colorbar(im, ax=ax, label="accuracy (%)", shrink=0.8)
    ax.set_title("Accuracy croisée entre groupes (diagonale = test interne)")
    return _save(fig, path)


# ---------------------------------------------------------------- analyse des signaux
def mean_signal(by_group, class_index, path, ref=None):
    """Signal moyen de chaque groupe sur les 6 axes (± 1 écart-type pour la référence).

    by_group = {groupe: signaux normalisés (n, 75, 6) de ce geste}.
    """
    colors = group_colors(list(by_group), ref)
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
    for a, ax in enumerate(axs.ravel()):
        for g, S in by_group.items():
            mu, sd = S[:, :, a].mean(0), S[:, :, a].std(0)
            t = np.arange(len(mu))
            ax.plot(t, mu, color=colors[g], lw=2.4 if g == ref else 1.4, label=g, zorder=3 if g == ref else 2)
            if g == ref:
                ax.fill_between(t, mu - sd, mu + sd, color=colors[g], alpha=0.15, lw=0)
        ax.set_title(CHANNELS[a], fontsize=10)
    for ax in axs[1]:
        ax.set_xlabel("échantillon")
    axs[0, 0].legend(frameon=False, fontsize=8)
    detail = f" ({ref} avec ±1 écart-type)" if ref else ""
    fig.suptitle(f"{CLASSES[class_index]} : signal moyen normalisé{detail}")
    fig.tight_layout()
    return _save(fig, path)


def circle_turns(circles, path, ref=None):
    """Répartition du nombre de tours par cercle, par groupe. circles = {groupe: (n, 75, 6)}."""
    colors = group_colors(list(circles), ref)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(circles)
    for i, (g, A) in enumerate(circles.items()):
        pct = analysis.turn_distribution(A)
        bars = ax.bar(np.arange(4) + (i - (len(circles) - 1) / 2) * width, pct, width * 0.95, color=colors[g], label=g)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, color=TEXT_COLOR)
    ax.set_xticks(range(4), ["1 tour\n(lent)", "2 tours", "3 tours", "4+\n(rapide)"])
    ax.set_ylabel("% des enregistrements circle")
    ax.set_title("Vitesse : nombre de cercles faits en 1,5 s")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def circle_direction(circles, path, ref=None):
    """Part des cercles faits dans le sens horaire (plan acc_x / acc_y), par groupe."""
    colors = group_colors(list(circles), ref)
    names = list(circles)
    pct = [np.mean(analysis.signed_area(A) < 0) * 100 for A in circles.values()]
    fig, ax = plt.subplots(figsize=(1.5 * len(names) + 2, 4))
    bars = ax.bar(range(len(names)), pct, 0.6, color=[colors[g] for g in names])
    ax.bar_label(bars, fmt="%.0f %%", fontsize=9, color=TEXT_COLOR)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% de cercles dans le sens horaire")
    ax.set_title("Sens de rotation (plan acc_x / acc_y)")
    return _save(fig, path)


def axis_correlations(by_class_group, path, ref=None):
    """Corrélation entre axes, une ligne par geste et une colonne par groupe.

    by_class_group = {geste: {groupe: signaux normalisés (n, 75, 6)}}.
    """
    classes = list(by_class_group)
    names = list(next(iter(by_class_group.values())))
    short = ["ax", "ay", "az", "gx", "gy", "gz"]
    fig, axs = plt.subplots(len(classes), len(names), figsize=(3.3 * len(names), 3.3 * len(classes)), squeeze=False)
    for k, c in enumerate(classes):
        ref_corr = analysis.mean_correlation(by_class_group[c][ref]) if ref in names else None
        for j, g in enumerate(names):
            C = analysis.mean_correlation(by_class_group[c][g])
            ax = axs[k, j]
            ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
            for a in range(6):
                for b in range(6):
                    ax.text(b, a, f"{C[a, b]:.1f}".replace("0.", ".").replace("-.0", ".0"), ha="center",
                            va="center", fontsize=7, color="white" if abs(C[a, b]) > 0.6 else "#1f1f1d")
            ax.set_xticks(range(6), short, fontsize=8)
            ax.set_yticks(range(6), short, fontsize=8)
            ax.grid(False)
            title = f"{g} - {c}"
            if ref_corr is not None and g != ref:
                title += f"\nécart vs {ref} = {np.abs(C - ref_corr).mean():.2f}"
            ax.set_title(title, fontsize=9, color=REF_COLOR if g == ref else "#1f1f1d")
    fig.suptitle("Corrélation entre axes (rouge = bougent ensemble, bleu = en opposition)")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- choix des axes
def axes_comparison(rows, path, holdout):
    """Meilleure combinaison pour chaque nombre d'axes. rows = [{n_axes, axes, acc_test, acc_holdout}]."""
    labels = [f"{r['n_axes']} axes\n{r['axes'].replace(' + ', '+')}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.6 * len(rows) + 2, 4.5))
    b1 = ax.bar(x - 0.2, [r["acc_test"] * 100 for r in rows], 0.38, color=OTHER_COLORS[0], label="test interne")
    b2 = ax.bar(x + 0.2, [r["acc_holdout"] * 100 for r in rows], 0.38, color=REF_COLOR, label=f"{holdout} (jamais vu)")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f", fontsize=9, color=TEXT_COLOR)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 128)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Meilleure combinaison d'axes (classée sur {holdout})")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    return _save(fig, path)
