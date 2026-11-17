"""Compare le modèle en float32, float16 et int8 : taille, vitesse, accuracy.

Utilise le modèle entraîné par train.py et le même jeu de test que lui.

    python scripts/quantize.py
    python scripts/quantize.py --model models/autre.pt

La vitesse dépend de la machine : sur un Mac, int8 peut être plus lent que
float32, car le modèle est minuscule et les conversions coûtent plus que le calcul.
"""
import argparse
import json
from pathlib import Path

import torch

from gestures.config import DATA_DIR, DEFAULT_MODEL, RESULTS_DIR
from gestures.data import load_recordings
from gestures.metrics import accuracy
from gestures.preprocessing import normalize, split
from gestures.quantization import Float16, size_kb, speed, to_int8
from gestures.training import load_model, to_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "quantization")
    args = parser.parse_args()

    model, ckpt = load_model(args.model)
    rec = load_recordings(args.data, verbose=False)
    X = to_tensor(normalize(rec.X))
    # même découpage que train.py, grâce à la graine enregistrée dans le modèle
    train, _, test = split(rec.y, rec.groups, holdout_group=ckpt.get("holdout_group"), seed=ckpt.get("split_seed", 0))

    versions = {"float32": model, "float16": Float16(model).eval(), "int8": to_int8(model, X[train])}
    rows = []
    print(f"{'version':8s} {'taille':>9s} {'vitesse':>12s} {'accuracy':>9s}")
    for name, m in versions.items():
        with torch.no_grad():
            acc = accuracy(rec.y[test], m(X[test]).argmax(1).numpy())
        row = {"version": name, "size_kb": size_kb(m), "predictions_per_s": speed(m, X[test][:1]), "accuracy": acc}
        rows.append(row)
        print(f"{name:8s} {row['size_kb']:6.1f} Ko {row['predictions_per_s']:8.0f} it/s {acc * 100:8.1f} %")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quantization.json").write_text(json.dumps(rows, indent=2))
    print(f"Résultats dans {args.out}/quantization.json")


if __name__ == "__main__":
    main()
