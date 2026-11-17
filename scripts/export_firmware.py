"""Transforme models/movements.pt en firmware/movement_classifier/model_weights.h.

    python scripts/export_firmware.py
    python scripts/export_firmware.py --model models/autre.pt --out /tmp/model_weights.h

À relancer après chaque entraînement, puis téléverser le sketch sur l'ESP32 :
sinon la carte garde les anciens poids.
"""
import argparse
from pathlib import Path

from gestures.config import DEFAULT_HEADER, DEFAULT_MODEL, ROOT
from gestures.export import check_firmware_compatible, write_header
from gestures.training import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_HEADER)
    args = parser.parse_args()

    model, ckpt = load_model(args.model)
    check_firmware_compatible(model, ckpt)
    source = args.model.resolve()
    source = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    n = write_header(model, ckpt["classes"], args.out, source=str(source))
    print(f"Écrit : {args.out}  ({n} poids, gestes : {', '.join(ckpt['classes'])})")


if __name__ == "__main__":
    main()
