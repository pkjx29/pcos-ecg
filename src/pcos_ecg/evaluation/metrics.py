"""Binary screening metrics with explicit handling of undefined values.

Positive class = 1 (PCOS). Undefined metrics (e.g. AUC on a single-class fold,
precision with no positive predictions) are NaN, never 0 or 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLD = 0.5  # fixed in advance; not tuned on any evaluation fold


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = THRESHOLD) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sens = _ratio(tp, tp + fn)
    spec = _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "f1": f1,
        "balanced_accuracy": float(np.nanmean([sens, spec])) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if both_classes else float("nan"),
    }


SUMMARY_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
)
