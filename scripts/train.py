"""Entraîne le modèle de la carte et mesure sa qualité.

    python scripts/train.py                          # tous les groupes, 70 / 15 / 15
    python scripts/train.py --holdout-group group3   # test = tout group3 (personne jamais vue)
    python scripts/train.py --test-data autre/dossier  # test supplémentaire sur un autre dataset

Écrit :
    models/movements.pt                      le modèle (lu par export_firmware.py)
    results/training/learning_curves.png     loss et accuracy à chaque époque
    results/training/confusion_matrix.png    erreurs sur le jeu de test
    results/training/metrics.json            accuracy, précision, rappel, F1
"""
import argparse
import json
from pathlib import Path

from gestures import plots
from gestures.config import DATA_DIR, DEFAULT_MODEL, RESULTS_DIR
from gestures.data import describe, load_recordings
from gestures.metrics import confusion_matrix, format_report, scores
from gestures.model import count_parameters
from gestures.preprocessing import normalize, split
from gestures.training import add_training_args, config_from_args, fit, predict, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="dossier des CSV (parcouru récursivement)")
    parser.add_argument("--holdout-group", help="groupe gardé entièrement pour le test, ex. group3")
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
    if args.holdout_group and args.holdout_group not in rec.group_names():
        raise SystemExit(f"{args.holdout_group} absent des données ({', '.join(rec.group_names())}).")
    X = normalize(rec.X)
    train, val, test = split(rec.y, rec.groups, holdout_group=args.holdout_group, seed=args.seed)
    print(f"Entraînement {len(train)} | validation {len(val)} | test {len(test)}")

    # 2. Entraînement
    model, history = fit(X[train], rec.y[train], X[val], rec.y[val], config, verbose=True)
    print(f"Modèle : CNN 1D {config.widths}, {count_parameters(model)} paramètres")

    # 3. Évaluation
    cm = confusion_matrix(rec.y[test], predict(model, X[test])[0])
    title = f"Test sur {args.holdout_group} (jamais vu)" if args.holdout_group else "Test interne"
    print("\n" + format_report(title, cm))
    metrics = {"test": {"title": title, **scores(cm), "confusion": cm.tolist()}}
    args.results.mkdir(parents=True, exist_ok=True)
    plots.learning_curves(history, args.results / "learning_curves.png")
    plots.confusion(cm, title, args.results / "confusion_matrix.png")

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
    save_model(args.model_out, model, config, holdout_group=args.holdout_group, split_seed=args.seed)
    print(f"\nModèle : {args.model_out}\nFigures et métriques : {args.results}/")
    print("Étape suivante : python scripts/export_firmware.py")


if __name__ == "__main__":
    main()
