"""
Evaluation metrics for the fraud detection model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold on validation set that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[min(best_idx, len(thresholds) - 1)])


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Fraction of actual fraud in the top-k highest-scored transactions."""
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_idx].mean())


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    k: int = 1000,
    split_name: str = "val",
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        f"{split_name}/roc_auc": roc_auc_score(y_true, y_prob),
        f"{split_name}/avg_precision": average_precision_score(y_true, y_prob),
        f"{split_name}/f1": f1_score(y_true, y_pred, zero_division=0),
        f"{split_name}/precision": precision_score(y_true, y_pred, zero_division=0),
        f"{split_name}/recall": recall_score(y_true, y_pred, zero_division=0),
        f"{split_name}/precision_at_{k}": precision_at_k(y_true, y_prob, k),
        f"{split_name}/threshold": threshold,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    for k, v in metrics.items():
        print(f"  {k:<40s} {v:.4f}")
