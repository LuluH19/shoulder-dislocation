"""Entraîne le modèle de la carte et mesure sa qualité.

    python scripts/train.py                          # tous les groupes, 70 / 15 / 15
    python scripts/train.py --holdout-group group3   # test = tout group3 (personne jamais vue)
    python scripts/train.py --test-data autre/dossier  # test supplémentaire sur un autre dataset
    python scripts/train.py --pretrain-group group3  # group3 seul d'abord, puis tous les groupes

Avec --pretrain-group, trois modèles sont comparés sur le même jeu de test :
    1. le groupe choisi seul,
    2. le groupe choisi, puis tous les groupes (repart des poids du 1) : c'est lui qui est sauvé,
    3. tous les groupes directement, comme sans l'option (référence).

Écrit :
    models/movements.pt                      le modèle (lu par export_firmware.py)
    results/training/learning_curves.png     loss et accuracy à chaque époque
    results/training/confusion_matrix.png    erreurs sur le jeu de test
    results/training/metrics.json            accuracy, précision, rappel, F1
    results/training/pretraining.png         comparaison des 3 modèles (avec --pretrain-group)
"""
import argparse
import dataclasses
import json
from pathlib import Path

from gestures import plots
from gestures.config import DATA_DIR, DEFAULT_MODEL, RESULTS_DIR
from gestures.data import describe, load_recordings
from gestures.metrics import accuracy, confusion_matrix, format_report, scores
from gestures.model import count_parameters
from gestures.preprocessing import normalize, split
from gestures.training import add_training_args, config_from_args, fit, predict, save_model


def pretrain_then_all(X, rec, train, val, test, group, config, pretrain_epochs):
    """Entraîne les 3 modèles décrits en haut du fichier et compare leurs scores sur le test.

    Le test n'est jamais utilisé pour entraîner : le groupe seul ne voit que sa part
    de l'entraînement et de la validation.
    """
    y = rec.y
    tr_g, va_g = train[rec.groups[train] == group], val[rec.groups[val] == group]

    print(f"\n1. {group} seul ({len(tr_g)} enregistrements, {pretrain_epochs} époques)")
    alone, _ = fit(X[tr_g], y[tr_g], X[va_g], y[va_g], dataclasses.replace(config, epochs=pretrain_epochs), verbose=True)

    print(f"\n2. {group}, puis tous les groupes ({len(train)} enregistrements, {config.epochs} époques)")
    model, history = fit(X[train], y[train], X[val], y[val], config, verbose=True, init_model=alone)

    print(f"\n3. Référence : tous les groupes directement ({config.epochs} époques)")
    direct, _ = fit(X[train], y[train], X[val], y[val], config)

    models = {f"{group} seul": alone, f"{group} puis tous": model, "tous directement": direct}
    test_groups = [g for g in rec.group_names() if (rec.groups[test] == g).any()]
    columns = {"test complet": test, **{g: test[rec.groups[test] == g] for g in test_groups}}
    table = {name: {col: accuracy(y[idx], predict(m, X[idx])[0]) for col, idx in columns.items()}
             for name, m in models.items()}

    print(f"\nAccuracy sur le même jeu de test ({len(test)} gestes, jamais vus à l'entraînement)")
    print(f"{'modèle':22s}" + "".join(f"{c:>14s}" for c in columns))
    for name, row in table.items():
        print(f"{name:22s}" + "".join(f"{v * 100:13.1f}%" for v in row.values()))
    counts = "  ".join(f"{c} {len(idx)}" for c, idx in columns.items())
    print(f"{'(nombre de gestes)':22s} {counts}")
    return model, history, table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="dossier des CSV (parcouru récursivement)")
    parser.add_argument("--holdout-group", help="groupe gardé entièrement pour le test, ex. group3")
    parser.add_argument("--pretrain-group", help="groupe appris seul d'abord, avant tous les groupes, ex. group3")
    parser.add_argument("--pretrain-epochs", type=int, default=30, help="époques sur le groupe seul (défaut 30)")
    parser.add_argument("--test-data", type=Path, help="autre dataset, utilisé seulement pour un test en plus")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "training")
    add_training_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    # 1. Données
    print(f"Chargement de {args.data}")
    rec = load_recordings(args.data)
    print(describe(rec))
    for option, group in (("--holdout-group", args.holdout_group), ("--pretrain-group", args.pretrain_group)):
        if group and group not in rec.group_names():
            raise SystemExit(f"{option} {group} : absent des données ({', '.join(rec.group_names())}).")
    if args.pretrain_group and args.pretrain_group == args.holdout_group:
        raise SystemExit("--pretrain-group et --holdout-group doivent être différents : on ne peut pas apprendre sur le groupe de test.")
    X = normalize(rec.X)
    train, val, test = split(rec.y, rec.groups, holdout_group=args.holdout_group, seed=args.seed)
    print(f"Entraînement {len(train)} | validation {len(val)} | test {len(test)}")

    # 2. Entraînement
    pretraining = None
    if args.pretrain_group:
        model, history, pretraining = pretrain_then_all(X, rec, train, val, test, args.pretrain_group, config, args.pretrain_epochs)
    else:
        model, history = fit(X[train], rec.y[train], X[val], rec.y[val], config, verbose=True)
    print(f"Modèle : CNN 1D {config.widths}, {count_parameters(model)} paramètres")

    # 3. Évaluation
    cm = confusion_matrix(rec.y[test], predict(model, X[test])[0])
    title = f"Test sur {args.holdout_group} (jamais vu)" if args.holdout_group else "Test interne"
    if args.pretrain_group:
        title += f", pré-entraîné sur {args.pretrain_group}"
    print("\n" + format_report(title, cm))
    metrics = {"test": {"title": title, **scores(cm), "confusion": cm.tolist()}}
    args.results.mkdir(parents=True, exist_ok=True)
    plots.learning_curves(history, args.results / "learning_curves.png")
    plots.confusion(cm, title, args.results / "confusion_matrix.png")
    if pretraining:
        metrics["pretraining"] = {"group": args.pretrain_group, "pretrain_epochs": args.pretrain_epochs, "accuracy": pretraining}
        plots.grouped_accuracy(pretraining, args.results / "pretraining.png",
                               f"{args.pretrain_group} d'abord, ou tous les groupes directement ?")

    if args.test_data:
        ext = load_recordings(args.test_data)
        shared = set(ext.group_names()) & set(rec.group_names())
        if shared:
            print(f"ATTENTION : groupes aussi présents à l'entraînement {sorted(shared)}, ce test n'est pas indépendant.")
        cm_ext = confusion_matrix(ext.y, predict(model, normalize(ext.X))[0])
        print("\n" + format_report(f"Test externe ({args.test_data})", cm_ext))
        metrics["external"] = {"data": str(args.test_data), **scores(cm_ext), "confusion": cm_ext.tolist()}
        plots.confusion(cm_ext, "Test externe", args.results / "confusion_matrix_external.png")

    # 4. Sauvegardes
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    save_model(args.model_out, model, config, holdout_group=args.holdout_group, split_seed=args.seed,
               pretrain_group=args.pretrain_group)
    print(f"\nModèle : {args.model_out}\nFigures et métriques : {args.results}/")
    print("Étape suivante : python scripts/export_firmware.py")


if __name__ == "__main__":
    main()
