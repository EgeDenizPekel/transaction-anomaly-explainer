"""
SHAP computation helpers for LightGBM.

Uses TreeExplainer which gives exact (not approximate) Shapley values.
"""

import numpy as np
import pandas as pd
import shap


# Columns that are metadata, not model features
NON_FEATURE_COLS = {"TransactionID", "TransactionDT", "isFraud"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Returns SHAP values array of shape (n_samples, n_features).
    For binary classification, LightGBM returns a list of two arrays (one per class).
    We take index [1] for the positive (fraud) class.
    """
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        return vals[1]
    return vals


def top_k_features(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    feature_values_row: np.ndarray,
    k: int = 3,
) -> list[dict]:
    """
    Return the top-k features by absolute SHAP value for a single transaction.

    Returns a list of dicts matching the API explanation contract:
        {
            "feature": str,
            "shap_value": float,       # signed SHAP value
            "feature_value": float,    # actual feature value
            "direction": str,          # "increases_risk" or "decreases_risk"
        }
    """
    abs_vals = np.abs(shap_values_row)
    top_indices = np.argsort(abs_vals)[::-1][:k]

    result = []
    for idx in top_indices:
        sv = float(shap_values_row[idx])
        result.append({
            "feature": feature_names[idx],
            "shap_value": round(sv, 4),
            "feature_value": round(float(feature_values_row[idx]), 4),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
        })
    return result


def explanation_contract(
    transaction_id,
    anomaly_score: float,
    threshold: float,
    shap_values_row: np.ndarray,
    feature_names: list[str],
    feature_values_row: np.ndarray,
    base_value: float,
    k: int = 3,
) -> dict:
    """Build the full per-transaction explanation dict."""
    score = round(float(anomaly_score), 4)
    return {
        "transaction_id": str(transaction_id),
        "anomaly_score": score,
        "is_flagged": score >= threshold,
        "alert_level": _alert_level(score),
        "top_features": top_k_features(shap_values_row, feature_names, feature_values_row, k),
        "base_value": round(float(base_value), 4),
    }


def _spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation using numpy (no scipy dependency)."""
    n = len(a)
    if n < 2:
        return 1.0
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    d = rank_a - rank_b
    denom = n * (n**2 - 1)
    if denom == 0:
        return 1.0
    return float(1 - 6 * np.sum(d**2) / denom)


def attribution_stability(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    feature_cols: list[str],
    n_perturbations: int = 5,
    noise_fraction: float = 0.05,
) -> float:
    """
    Measure SHAP attribution stability under small feature perturbations.

    Perturbs each numeric feature by Gaussian noise (std = noise_fraction * |value|),
    re-computes SHAP, and measures Spearman rank correlation between original and
    perturbed feature importances. Returns the mean rank correlation.

    Range: 0.0 (completely unstable) to 1.0 (perfectly stable).
    High stability (>= 0.85) means the top-3 ranking is robust to small input changes.
    Low stability (<= 0.6) means the explanation is sensitive to minor feature variations
    and should be interpreted with caution.

    Note: This adds multiple SHAP evaluations per transaction - use only when the
    stability information is needed (e.g. the constrained v2/v3 prompt is active).
    """
    original_shap = compute_shap_values(explainer, X)[0]
    original_importance = np.abs(original_shap)

    rng = np.random.default_rng(42)
    correlations = []

    for _ in range(n_perturbations):
        X_perturbed = X.copy()
        for col in feature_cols:
            val = float(X_perturbed[col].iloc[0])
            if val not in (-999.0, 0.0):
                noise = rng.normal(0, abs(val) * noise_fraction + 1e-6)
                X_perturbed[col] = val + noise

        perturbed_shap = compute_shap_values(explainer, X_perturbed)[0]
        perturbed_importance = np.abs(perturbed_shap)

        corr = _spearman_rank_corr(original_importance, perturbed_importance)
        if not np.isnan(corr):
            correlations.append(corr)

    return round(float(np.mean(correlations)) if correlations else 0.0, 4)


def _alert_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"
