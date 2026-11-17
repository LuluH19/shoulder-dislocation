"""Le code C de la carte doit calculer exactement comme PyTorch.

On exporte un modèle, on compile firmware/movement_classifier/cnn.h avec un petit
programme de test, et on compare les probabilités sur des fenêtres aléatoires.
Ignoré si aucun compilateur C n'est disponible.
"""
import shutil
import subprocess

import numpy as np
import pytest
import torch

from gestures.config import CLASSES, FIRMWARE_DIR
from gestures.export import fuse_conv_bn, write_header
from gestures.model import CNN1D
from gestures.preprocessing import normalize
from gestures.training import predict

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")

HARNESS = r"""
#include <stdio.h>
#include "cnn.h"
int main(void) {
  float x[N_CH * N_SAMPLES], probs[N_CLASSES];
  while (1) {
    for (int i = 0; i < N_CH * N_SAMPLES; i++)
      if (scanf("%f", &x[i]) != 1) return 0;
    normalize_window(x);
    cnn_predict(x, probs);
    for (int k = 0; k < N_CLASSES; k++) printf("%.7f ", probs[k]);
    printf("\n");
  }
}
"""


def trained_like_model():
    """Un modèle aux BatchNorm non triviales, comme après un entraînement."""
    torch.manual_seed(0)
    model = CNN1D()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    return model.eval()


def test_fused_convolution_matches_conv_then_batchnorm():
    model = trained_like_model()
    conv, bn = model.features[0], model.features[1]
    w, b = fuse_conv_bn(conv, bn)
    x = torch.randn(3, 6, 75)
    expected = bn(conv(x))
    got = torch.nn.functional.conv1d(x, w, b, padding=2)
    assert torch.allclose(got, expected, atol=1e-5)


@pytest.mark.skipif(CC is None, reason="pas de compilateur C")
def test_c_firmware_matches_pytorch(tmp_path):
    model = trained_like_model()
    write_header(model, CLASSES, tmp_path / "model_weights.h", source="test")
    shutil.copy(FIRMWARE_DIR / "cnn.h", tmp_path / "cnn.h")
    (tmp_path / "main.c").write_text(HARNESS)
    subprocess.run([CC, "-O2", "-o", str(tmp_path / "cnn"), str(tmp_path / "main.c"), "-lm"], check=True)

    raw = np.random.default_rng(0).normal(size=(20, 75, 6)).astype(np.float32) * [1, 1, 1, 100, 100, 100]
    # le firmware range la fenêtre en [axe][temps]
    stdin = "\n".join(" ".join(f"{v:.8e}" for v in w.T.ravel()) for w in raw)
    out = subprocess.run([str(tmp_path / "cnn")], input=stdin, capture_output=True, text=True, check=True).stdout
    c_probs = np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])

    _, torch_probs = predict(model, normalize(raw))
    assert np.allclose(c_probs, torch_probs, atol=1e-4)
