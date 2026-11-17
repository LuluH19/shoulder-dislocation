"""Le modèle marche-t-il sur une personne qu'il n'a jamais vue ?

Deux expériences :
  1. Un modèle par groupe, testé sur son propre test puis sur chaque autre groupe
     (matrice croisée).
  2. Un modèle entraîné sur tous les groupes sauf un, testé sur celui-là
     (leave-one-group-out) : la mesure la plus honnête pour une nouvelle personne.

    python scripts/compare_groups.py
    python scripts/compare_groups.py --ref group2 --epochs 30

Écrit dans results/groups/ :
    01_accuracy_modele_<ref>.png, 02_confusion_modele_<ref>.png,
    03_matrice_croisee.png, 07_groupe_jamais_vu.png, comparaison_groupes.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gestures import plots
from gestures.config import DATA_DIR, RESULTS_DIR
from gestures.data import describe, load_recordings
from gestures.metrics import accuracy, confusion_matrix, scores
from gestures.preprocessing import normalize, split
from gestures.training import add_training_args, config_from_args, fit, predict


def cross_group(X, rec, names, config):
    """results[train][test] = (vrais gestes, gestes prédits)."""
    results = {}
    for g_train in names:
        idx = np.flatnonzero(rec.groups == g_train)
        tr, va, te = split(rec.y[idx], val_frac=0.15, test_frac=0.2, seed=config.seed)
        tr, va, te = idx[tr], idx[va], idx[te]
        print(f"  modèle {g_train} ({len(tr)} enregistrements d'entraînement)")
        model, _ = fit(X[tr], rec.y[tr], X[va], rec.y[va], config)
        results[g_train] = {}
        for g_test in names:
            ids = te if g_test == g_train else np.flatnonzero(rec.groups == g_test)
            results[g_train][g_test] = (rec.y[ids], predict(model, X[ids])[0])
    return results


def leave_one_group_out(X, rec, names, config):
    """{groupe jamais vu: (vrais gestes, gestes prédits)} pour un modèle entraîné sur tous les autres."""
    results = {}
    for g in names:
        tr, va, te = split(rec.y, rec.groups, holdout_group=g, seed=config.seed)
        print(f"  sans {g} ({len(tr)} enregistrements d'entraînement)")
        model, _ = fit(X[tr], rec.y[tr], X[va], rec.y[va], config)
        results[g] = (rec.y[te], predict(model, X[te])[0])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--ref", default="group3", help="groupe mis en avant dans les figures 01 et 02")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "groups")
    add_training_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    rec = load_recordings(args.data)
    print(describe(rec))
    names = rec.group_names()
    if args.ref not in names:
        raise SystemExit(f"{args.ref} absent des données ({', '.join(names)}).")
    names = [args.ref] + [g for g in names if g != args.ref]
    X = normalize(rec.X)

    print("\n1. Un modèle par groupe")
    cross = cross_group(X, rec, names, config)
    M = np.array([[accuracy(*cross[a][b]) * 100 for b in names] for a in names])
    print(f"\n{'entraîné sur':14s}" + "".join(f"{g:>9s}" for g in names))
    for a, row in zip(names, M):
        print(f"{a:14s}" + "".join(f"{v:8.0f}%" for v in row))

    print("\n2. Un groupe jamais vu")
    logo = leave_one_group_out(X, rec, names, config)
    for g, (y, p) in logo.items():
        print(f"  {g:8s} {accuracy(y, p) * 100:5.1f} %")

    print(f"\nFigures dans {args.out}/")
    colors = plots.group_colors(names, args.ref)
    ref_scores = [scores(confusion_matrix(*cross[args.ref][g])) for g in names]
    plots.accuracy_bars(
        names, [s["accuracy"] * 100 for s in ref_scores], args.out / f"01_accuracy_modele_{args.ref}.png",
        f"Modèle {args.ref} testé sur chaque groupe", colors=[colors[g] for g in names],
        labels=[f"{s['accuracy'] * 100:.0f} %\nF1 {s['f1_macro'] * 100:.0f}" for s in ref_scores],
        xlabels=[f"{g}\n({'test' if g == args.ref else 'jamais vu'})" for g in names])
    plots.confusion_grid(
        [(f"Modèle {args.ref} -> {g}\naccuracy = {accuracy(*cross[args.ref][g]):.3f}",
          confusion_matrix(*cross[args.ref][g])) for g in names],
        args.out / f"02_confusion_modele_{args.ref}.png")
    plots.cross_accuracy(M, names, args.out / "03_matrice_croisee.png", ref=args.ref)
    plots.accuracy_bars(
        names, [accuracy(*logo[g]) * 100 for g in names], args.out / "07_groupe_jamais_vu.png",
        "Entraîné sur les autres groupes, testé sur celui-ci", colors=[colors[g] for g in names])

    summary = {
        "config": vars(args) | {"data": str(args.data), "out": str(args.out)},
        "cross_accuracy": {a: {b: float(M[i, j] / 100) for j, b in enumerate(names)} for i, a in enumerate(names)},
        "leave_one_group_out": {g: scores(confusion_matrix(*logo[g])) for g in names},
    }
    (args.out / "comparaison_groupes.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
