"""Faut-il vraiment les 6 axes du capteur ?

Entraîne un modèle pour chaque combinaison de 2, 4 et 6 axes (15 + 15 + 1 = 31
entraînements). Un groupe entier est gardé de côté pour mesurer ce que donne
chaque combinaison sur une personne jamais vue.

    python scripts/compare_axes.py                         # group2 jamais vu, 30 époques
    python scripts/compare_axes.py --holdout-group group5 --sizes 2 4

Écrit dans results/axes/ :
    combinaisons_axes.csv     toutes les combinaisons et leurs scores
    meilleurs_axes.png        meilleure combinaison pour chaque nombre d'axes
    confusion_meilleurs.png   leurs matrices de confusion sur le groupe jamais vu
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from gestures import plots
from gestures.config import CHANNELS, DATA_DIR, RESULTS_DIR
from gestures.data import load_recordings
from gestures.metrics import accuracy, confusion_matrix
from gestures.preprocessing import normalize, split
from gestures.training import add_training_args, config_from_args, fit, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--holdout-group", default="group2", help="groupe jamais vu à l'entraînement")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6], help="nombres d'axes à essayer")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "axes")
    add_training_args(parser, epochs=30)
    args = parser.parse_args()
    config = config_from_args(args)

    rec = load_recordings(args.data)
    if args.holdout_group not in rec.group_names():
        raise SystemExit(f"{args.holdout_group} absent des données ({', '.join(rec.group_names())}).")
    X = normalize(rec.X)
    # entraînement / validation / test interne sur les autres groupes ; le groupe gardé à part sert de 2e test
    held = np.flatnonzero(rec.groups == args.holdout_group)
    others = np.flatnonzero(rec.groups != args.holdout_group)
    tr, va, te = (others[i] for i in split(rec.y[others], seed=args.seed))
    print(f"Entraînement {len(tr)} | validation {len(va)} | test interne {len(te)} | {args.holdout_group} {len(held)}")

    rows = []
    for k in args.sizes:
        for combo in combinations(range(len(CHANNELS)), k):
            axes = list(combo)
            model, _ = fit(X[tr][:, :, axes], rec.y[tr], X[va][:, :, axes], rec.y[va], config)
            p_test = predict(model, X[te][:, :, axes])[0]
            p_held = predict(model, X[held][:, :, axes])[0]
            rows.append({"n_axes": k, "axes": " + ".join(CHANNELS[i] for i in axes), "idx": axes,
                         "acc_test": accuracy(rec.y[te], p_test), "acc_holdout": accuracy(rec.y[held], p_held),
                         "cm_holdout": confusion_matrix(rec.y[held], p_held)})
            print(f"  {rows[-1]['axes']:48s} test {rows[-1]['acc_test'] * 100:5.1f} %"
                  f"  {args.holdout_group} {rows[-1]['acc_holdout'] * 100:5.1f} %")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "combinaisons_axes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_axes", "axes", "acc_test_interne", f"acc_{args.holdout_group}"])
        for r in rows:
            w.writerow([r["n_axes"], r["axes"], f"{r['acc_test']:.4f}", f"{r['acc_holdout']:.4f}"])

    best = [max((r for r in rows if r["n_axes"] == k), key=lambda r: r["acc_holdout"]) for k in args.sizes]
    print(f"\nMeilleure combinaison (classée sur {args.holdout_group}) :")
    for r in best:
        print(f"  {r['n_axes']} axes : {r['axes']:48s} test {r['acc_test'] * 100:5.1f} %"
              f"  {args.holdout_group} {r['acc_holdout'] * 100:5.1f} %")
    plots.axes_comparison(best, args.out / "meilleurs_axes.png", args.holdout_group)
    plots.confusion_grid([(f"{r['n_axes']} axes : {r['axes']}\n{args.holdout_group} = {r['acc_holdout']:.3f}",
                           r["cm_holdout"]) for r in best], args.out / "confusion_meilleurs.png")
    print(f"Résultats dans {args.out}/")


if __name__ == "__main__":
    main()
