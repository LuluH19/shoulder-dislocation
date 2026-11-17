import numpy as np
import torch

from gestures.preprocessing import augment, normalize, split


def test_normalize_gives_zero_mean_unit_std_per_axis():
    X = np.random.default_rng(0).normal(5, 3, size=(4, 75, 6)) * np.arange(1, 7)
    Z = normalize(X)
    assert np.allclose(Z.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(Z.std(axis=1), 1, atol=1e-4)


def test_normalize_ignores_amplitude_and_offset():
    x = np.random.default_rng(1).normal(size=(75, 6))
    assert np.allclose(normalize(x), normalize(10 * x + 3), atol=1e-5)


def test_split_has_no_overlap_and_is_balanced():
    y = np.repeat([0, 1, 2], 100)
    train, val, test = split(y, val_frac=0.15, test_frac=0.15, seed=0)
    assert len(set(train) | set(val) | set(test)) == len(y)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert [int((y[test] == c).sum()) for c in range(3)] == [15, 15, 15]


def test_split_with_holdout_group_tests_on_that_group_only():
    y = np.tile([0, 1, 2], 40)
    groups = np.array(["group1"] * 60 + ["group2"] * 60)
    train, val, test = split(y, groups, holdout_group="group2", seed=0)
    assert set(groups[test]) == {"group2"}
    assert set(groups[np.concatenate([train, val])]) == {"group1"}
    assert len(test) == 60


def test_augment_keeps_shape():
    x = torch.randn(6, 75)
    assert augment(x).shape == x.shape
