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


def _alert_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"
