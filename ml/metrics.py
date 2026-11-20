"""Holdout assessment and validation-only decision threshold selection."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Labels and scores must be non-empty, equal length and finite")
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    result = {
        "n": int(len(labels)), "threshold": round(float(threshold), 5),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
    }
    result["roc_auc"] = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    result["average_precision"] = float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None
    return result


def find_threshold(labels, scores, max_fpr=0.05):
    """Prefer best F1 among thresholds meeting target FPR; fallback to lowest FPR.

    Only call with VALIDATION data; never tune on the held-out test set.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Threshold selection requires both classes in validation")
    candidates = np.unique(np.r_[0.0, scores, np.nextafter(scores, np.inf), 1.0])
    candidates = candidates[(candidates >= 0) & (candidates <= 1)]
    results = [(float(t), evaluate(labels, scores, t)) for t in candidates]
    feasible = [entry for entry in results if entry[1]["false_positive_rate"] <= max_fpr]
    if feasible:
        return max(feasible, key=lambda item: (item[1]["f1"], -item[0]))[0]
    return min(results, key=lambda item: (item[1]["false_positive_rate"], -item[1]["f1"]))[0]


def source_report(labels, scores, sources, threshold):
    report = {}
    for source in sorted(set(sources)):
        indices = [i for i, item in enumerate(sources) if item == source]
        report[source] = evaluate([labels[i] for i in indices], [scores[i] for i in indices], threshold)
    return report
