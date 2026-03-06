"""
Model probability calibration metrics.

Brier score: mean squared error of predicted probabilities vs actual labels.
  - Range: 0.0 (perfect) to 1.0 (worst).
  - Naive baseline (always predict class prior p): Brier = p * (1 - p).
  - For 3.5% fraud base rate, naive baseline = ~0.034.
  - A useful model must beat this; a well-calibrated model will be close to but
    below the naive baseline when fraud is rare.

Reliability diagram: predicted probability vs actual fraud rate per bin.
  A perfectly calibrated model lies on the diagonal (predicted = actual).
"""

import numpy as np
from sklearn.metrics import brier_score_loss


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score. Lower is better."""
    return float(brier_score_loss(y_true, y_prob))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Bin predictions and return (mean_predicted, actual_rate, count) per bin.
    Used for reliability (calibration) diagrams. Only returns non-empty bins.

    Args:
        y_true:  Binary labels (0/1).
        y_prob:  Predicted probabilities in [0, 1].
        n_bins:  Number of equal-width probability bins.

    Returns:
        List of dicts with keys: mean_predicted, actual_rate, count, bin_lower, bin_upper.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    result = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        if count == 0:
            continue
        result.append({
            "mean_predicted": round(float(y_prob[mask].mean()), 4),
            "actual_rate": round(float(y_true[mask].mean()), 4),
            "count": count,
            "bin_lower": round(float(bins[i]), 2),
            "bin_upper": round(float(bins[i + 1]), 2),
        })
    return result
