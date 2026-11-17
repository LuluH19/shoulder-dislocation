"""Pourquoi les groupes ne font-ils pas le même geste ? (aucun entraînement)

Compare les signaux des groupes : forme moyenne, vitesse et sens des cercles,
orientation du capteur (corrélation entre axes).

    python scripts/analyze_groups.py
    python scripts/analyze_groups.py --ref group2

Écrit dans results/groups/ :
    04_signal_moyen_<geste>.png   forme moyenne de chaque geste, par groupe
    05_vitesse_circle.png         nombre de tours par cercle
    06_correlation_axes.png       quels axes bougent ensemble
    08_sens_circle.png            sens de rotation des cercles
"""
import argparse
from pathlib import Path

from gestures import plots
from gestures.config import CLASSES, DATA_DIR, RESULTS_DIR
from gestures.data import describe, load_recordings
from gestures.preprocessing import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--ref", default="group3", help="groupe mis en avant (orange)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "groups")
    args = parser.parse_args()

    rec = load_recordings(args.data)
    print(describe(rec))
    names = rec.group_names()
    ref = args.ref if args.ref in names else None
    if ref:
        names = [ref] + [g for g in names if g != ref]
    X = normalize(rec.X)

    # by_class[geste][groupe] = signaux normalisés de ce geste dans ce groupe
    by_class = {c: {g: X[(rec.y == k) & (rec.groups == g)] for g in names} for k, c in enumerate(CLASSES)}

    print(f"Figures dans {args.out}/")
    for k, c in enumerate(CLASSES):
        print("  ", plots.mean_signal(by_class[c], k, args.out / f"04_signal_moyen_{c}.png", ref=ref))
    print("  ", plots.circle_turns(by_class["circle"], args.out / "05_vitesse_circle.png", ref=ref))
    print("  ", plots.axis_correlations(by_class, args.out / "06_correlation_axes.png", ref=ref))
    print("  ", plots.circle_direction(by_class["circle"], args.out / "08_sens_circle.png", ref=ref))


if __name__ == "__main__":
    main()
