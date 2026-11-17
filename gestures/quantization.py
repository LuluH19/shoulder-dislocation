"""Compresser le modèle : float16 (2 octets par poids) et int8 (1 octet)."""
import copy
import io
import time
import warnings

import torch
from torch import nn


def size_kb(model):
    """Taille des poids sauvegardés, en Ko."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024


@torch.no_grad()
def speed(model, x, n=2000):
    """Prédictions d'un seul enregistrement par seconde (dépend de la machine)."""
    for _ in range(50):  # échauffement
        model(x)
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    return n / (time.perf_counter() - t)


class Float16(nn.Module):
    """Même modèle, poids et calculs en demi-précision."""

    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model).half()

    def forward(self, x):
        return self.model(x.half()).float()


@torch.no_grad()
def to_int8(model, calibration):
    """Quantization statique PyTorch : poids et calculs en entiers 8 bits.

    `calibration` (n, axes, 75) sert à fixer les échelles des activations.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "qnnpack"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(copy.deepcopy(model).eval(), get_default_qconfig_mapping("qnnpack"),
                              example_inputs=(calibration[:1],))
        prepared(calibration)
        return convert_fx(prepared)
