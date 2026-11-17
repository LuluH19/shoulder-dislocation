"""Mesurer la qualité d'un modèle : matrice de confusion, accuracy, précision, rappel, F1."""
import numpy as np

from .config import CLASSES


def confusion_matrix(y_true, y_pred, n_classes=len(CLASSES)):
    """cm[vrai, prédit] = nombre d'enregistrements."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def scores(cm):
    """Accuracy globale et, pour chaque geste, précision / rappel / F1 (+ moyennes macro)."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)   # combien de fois chaque geste a été prédit
    actual = cm.sum(axis=1)      # combien de vrais exemples de chaque geste
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "n": int(cm.sum()),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {c: {"precision": float(precision[k]), "recall": float(recall[k]), "f1": float(f1[k]),
                          "correct": int(tp[k]), "total": int(actual[k])}
                      for k, c in enumerate(CLASSES)},
    }


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def format_report(title, cm):
    """Texte lisible pour le terminal."""
    s = scores(cm)
    lines = [f"{title} : accuracy {s['accuracy'] * 100:.1f} % ({np.trace(cm)}/{s['n']})",
             "  matrice de confusion (lignes = vrai geste, colonnes = prédit) :"]
    width = max(len(c) for c in CLASSES)
    lines.append("  " + " " * width + "".join(f"{c[:5]:>7s}" for c in CLASSES))
    for k, c in enumerate(CLASSES):
        lines.append(f"  {c:{width}s}" + "".join(f"{v:7d}" for v in cm[k]))
    for c, m in s["per_class"].items():
        lines.append(f"  {c:{width}s}  précision {m['precision'] * 100:5.1f} %  rappel {m['recall'] * 100:5.1f} %"
                     f"  F1 {m['f1'] * 100:5.1f} %")
    return "\n".join(lines)
