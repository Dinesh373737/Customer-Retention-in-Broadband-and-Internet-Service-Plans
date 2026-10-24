"""Metrics and decision thresholds; callers supply the proper data partition."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def evaluate_probabilities(y_true, probability, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) != len(probability) or not len(y_true):
        raise ValueError("Labels and probabilities must have equal nonzero length.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite numbers between zero and one.")
    predicted = probability >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else np.nan,
        # Average precision is the non-interpolated area summary used for PR-AUC.
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def validation_threshold(y_true, probability) -> float:
    """Choose the highest threshold among tied validation F1 maxima."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(scores, np.max(scores)))
    return float(thresholds[best[-1]])
