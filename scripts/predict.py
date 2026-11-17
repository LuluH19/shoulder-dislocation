"""Prédit le geste d'un ou plusieurs fichiers CSV avec le modèle entraîné.

    python scripts/predict.py data/group3/circle/3_Circle_1.csv
    python scripts/predict.py data/group2/vertical/*.csv --model models/movements.pt
"""
import argparse
from pathlib import Path

from gestures.config import DEFAULT_MODEL
from gestures.data import read_csv, resample
from gestures.preprocessing import normalize
from gestures.training import load_model, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    model, ckpt = load_model(args.model)
    classes = ckpt["classes"]
    for path in args.csv:
        signal = read_csv(path)
        if signal is None:
            print(f"{path} : format non reconnu")
            continue
        pred, probs = predict(model, normalize(resample(signal))[None])
        detail = "  ".join(f"{c} {p * 100:5.1f} %" for c, p in zip(classes, probs[0]))
        print(f"{path} -> {classes[pred[0]]:10s} [{detail}]")


if __name__ == "__main__":
    main()
