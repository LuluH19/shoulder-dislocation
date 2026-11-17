"""Le réseau de neurones : un CNN à une dimension."""
from torch import nn

from .config import CHANNELS, CLASSES, MODEL_WIDTHS


class CNN1D(nn.Module):
    """3 convolutions qui glissent sur le temps, une moyenne, puis un score par geste.

    Entrée : (batch, 6 axes, 75 mesures). Sortie : (batch, 3) scores (logits).
    Avec les largeurs par défaut (16, 32, 64) : 9 715 paramètres.
    """

    def __init__(self, n_channels=len(CHANNELS), n_classes=len(CLASSES), widths=MODEL_WIDTHS, dropout=0.3):
        super().__init__()
        w1, w2, w3 = widths
        self.widths = tuple(widths)
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, w1, 5, padding=2), nn.BatchNorm1d(w1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(w1, w2, 5, padding=2), nn.BatchNorm1d(w2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(w2, w3, 3, padding=1), nn.BatchNorm1d(w3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # moyenne sur le temps : peu importe QUAND le motif apparaît
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(w3, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
