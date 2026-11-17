"""Exporter le modèle vers le firmware : écrit model_weights.h pour cnn.h.

Chaque BatchNorm est fusionnée dans la convolution qui la précède (même calcul
en mode évaluation, une couche de moins en C). Le Dropout ne sert qu'à
l'entraînement et disparaît.
"""
from pathlib import Path

import torch
from torch import nn

from .config import CHANNELS

FIRMWARE_WIDTHS = (16, 32, 64)  # tailles codées en dur dans firmware/movement_classifier/cnn.h


def fuse_conv_bn(conv, bn):
    """Renvoie (poids, biais) d'une convolution équivalente à conv suivie de bn."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale[:, None, None]
    bias = (conv.bias - bn.running_mean) * scale + bn.bias
    return weight.detach(), bias.detach()


def fused_layers(model):
    """[(poids, biais)] des 3 convolutions fusionnées puis de la couche finale."""
    layers = list(model.features)
    convs = [(m, layers[i + 1]) for i, m in enumerate(layers)
             if isinstance(m, nn.Conv1d) and isinstance(layers[i + 1], nn.BatchNorm1d)]
    linear = next(m for m in model.classifier if isinstance(m, nn.Linear))
    return [fuse_conv_bn(c, b) for c, b in convs] + [(linear.weight.detach(), linear.bias.detach())]


def c_array(name, tensor):
    """Un tenseur sous forme de tableau C (static const float ...)."""
    values = tensor.flatten().tolist()
    rows = [", ".join(f"{v:.8e}f" for v in values[i:i + 8]) for i in range(0, len(values), 8)]
    dims = "x".join(str(d) for d in tensor.shape)
    return f"// {dims}\nstatic const float {name}[{len(values)}] = {{\n  " + ",\n  ".join(rows) + "\n};\n"


def check_firmware_compatible(model, ckpt):
    if tuple(model.widths) != FIRMWARE_WIDTHS:
        raise SystemExit(f"cnn.h attend des convolutions {FIRMWARE_WIDTHS}, le modèle a {tuple(model.widths)}.")
    if len(ckpt.get("channels", CHANNELS)) != 6:
        raise SystemExit("cnn.h attend les 6 axes du capteur.")
    if ckpt.get("norm", "record") != "record":
        raise SystemExit("cnn.h suppose la normalisation 'record' (z-score par fenêtre).")


def write_header(model, classes, path, source="un modèle PyTorch"):
    """Écrit model_weights.h et renvoie le nombre de poids exportés."""
    (w1, b1), (w2, b2), (w3, b3), (wf, bf) = fused_layers(model)
    lines = [
        f"// Généré par scripts/export_firmware.py depuis {source} : ne pas modifier à la main.",
        "#pragma once",
        "",
        f"#define N_CLASSES {len(classes)}",
        "static const char* const CLASS_NAMES[N_CLASSES] = {" + ", ".join(f'"{c}"' for c in classes) + "};",
        "",
    ]
    # préfixe CNN_ : Arduino définit déjà B1, B10... (binary.h)
    tensors = {"W1": w1, "B1": b1, "W2": w2, "B2": b2, "W3": w3, "B3": b3, "WF": wf, "BF": bf}
    lines += [c_array("CNN_" + name, t) for name, t in tensors.items()]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return sum(t.numel() for t in tensors.values())
