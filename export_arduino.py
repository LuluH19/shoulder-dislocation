"""Exporte models/movements.pt en en-tête C (model_weights.h) pour le sketch Arduino.

Les BatchNorm sont fusionnées dans les convolutions (mode évaluation), le Dropout est ignoré.

Utilisation :
    .venv/bin/python export_arduino.py
    .venv/bin/python export_arduino.py --model models/movements.pt --out arduino/movement_classifier/model_weights.h
"""
import argparse
import os

import torch

EPS = 1e-5  # eps par défaut de nn.BatchNorm1d


def fuse(sd, conv, bn):
    w, b = sd[f"features.{conv}.weight"], sd[f"features.{conv}.bias"]
    g, beta = sd[f"features.{bn}.weight"], sd[f"features.{bn}.bias"]
    mean, var = sd[f"features.{bn}.running_mean"], sd[f"features.{bn}.running_var"]
    scale = g / torch.sqrt(var + EPS)
    return w * scale[:, None, None], (b - mean) * scale + beta


def c_array(name, t):
    vals = t.flatten().tolist()
    lines = [", ".join(f"{v:.8e}f" for v in vals[i:i + 8]) for i in range(0, len(vals), 8)]
    dims = "x".join(str(d) for d in t.shape)
    return f"// {dims}\nstatic const float {name}[{len(vals)}] = {{\n  " + ",\n  ".join(lines) + "\n};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/movements.pt")
    ap.add_argument("--out", default="arduino/movement_classifier/model_weights.h")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    if ckpt.get("norm") != "record":
        raise SystemExit("Ce sketch suppose --norm record (z-score par fenêtre).")
    sd = ckpt["state_dict"]
    w1, b1 = fuse(sd, 0, 1)
    w2, b2 = fuse(sd, 4, 5)
    w3, b3 = fuse(sd, 8, 9)
    wf, bf = sd["classifier.2.weight"], sd["classifier.2.bias"]

    classes = ckpt["classes"]
    out = [
        "// Généré par export_arduino.py depuis " + args.model + " : ne pas modifier à la main.",
        "#pragma once",
        "",
        f"#define N_CLASSES {len(classes)}",
        "static const char* const CLASS_NAMES[N_CLASSES] = {"
        + ", ".join(f'"{c}"' for c in classes) + "};",
        "",
    ]
    # préfixe CNN_ : Arduino définit déjà B1, B10... (binary.h)
    for name, t in (("W1", w1), ("B1", b1), ("W2", w2), ("B2", b2), ("W3", w3), ("B3", b3),
                    ("WF", wf), ("BF", bf)):
        out.append(c_array("CNN_" + name, t))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out))
    print(f"Écrit : {args.out}  (classes : {classes})")


if __name__ == "__main__":
    main()
