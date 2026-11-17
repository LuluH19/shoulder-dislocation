import numpy as np
import torch

from gestures.model import CNN1D, count_parameters
from gestures.training import TrainConfig, fit, load_model, predict, save_model


def test_board_model_size_and_output_shape():
    model = CNN1D()
    assert count_parameters(model) == 9715
    assert model(torch.randn(2, 6, 75)).shape == (2, 3)


def test_fit_learns_an_easy_problem_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 75)
    shapes = [np.sin(t), np.sign(np.sin(t)), np.sin(2 * t)]  # 3 "gestes" faciles à séparer
    X = np.stack([np.tile(shapes[k][:, None], (1, 6)) + 0.1 * rng.normal(size=(75, 6))
                  for k in range(3) for _ in range(20)]).astype(np.float32)
    y = np.repeat([0, 1, 2], 20)
    model, history = fit(X, y, X, y, TrainConfig(epochs=15, seed=0))
    assert (predict(model, X)[0] == y).mean() > 0.9
    assert len(history["val_loss"]) == 15

    path = tmp_path / "m.pt"
    save_model(path, model, TrainConfig())
    loaded, ckpt = load_model(path)
    assert ckpt["norm"] == "record"
    assert np.allclose(predict(loaded, X)[1], predict(model, X)[1], atol=1e-6)
