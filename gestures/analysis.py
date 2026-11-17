"""Mesures sur les signaux eux-mêmes (sans réseau de neurones).

Servent à comprendre POURQUOI les groupes ne font pas le même geste.
Toutes les fonctions prennent des signaux normalisés (n, 75, 6).
"""
import numpy as np


def count_turns(A):
    """Nombre de tours de chaque cercle en 1,5 s.

    On prend la fréquence la plus forte de acc_x + acc_y (FFT) :
    1 tour en 1,5 s = 0,67 Hz, 2 tours = 1,33 Hz, 3 tours = 2 Hz.
    """
    power = (np.abs(np.fft.rfft(A[:, :, :2], axis=1)) ** 2).sum(2)
    return power[:, 1:8].argmax(1) + 1


def turn_distribution(A):
    """% d'enregistrements faisant 1, 2, 3, puis 4 tours ou plus."""
    n = count_turns(A)
    return [np.mean(n == t) * 100 for t in (1, 2, 3)] + [np.mean(n >= 4) * 100]


def smooth(A, k=7):
    """Moyenne glissante sur k mesures : enlève le bruit, garde la forme."""
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda s: np.convolve(s, kernel, mode="valid"), 1, A)


def signed_area(A, i=0, j=1):
    """Aire signée du tracé (axe i, axe j) : < 0 sens horaire, > 0 anti-horaire."""
    A = smooth(A)
    x, y = A[:, :, i], A[:, :, j]
    return 0.5 * np.sum(x[:, :-1] * y[:, 1:] - x[:, 1:] * y[:, :-1], axis=1)


def mean_correlation(A):
    """Corrélation moyenne entre les 6 axes (6, 6).

    Si le capteur est tourné ou fixé autrement, les signes des corrélations changent.
    Les signaux étant en z-score, la corrélation est la moyenne des produits.
    """
    return np.einsum("nta,ntb->ab", A, A) / (A.shape[0] * A.shape[1])
