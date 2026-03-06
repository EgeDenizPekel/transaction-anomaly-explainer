"""
Single-feature perturbation counterfactual explanations.

For each feature that increases risk (positive SHAP), find the minimum reduction
in that feature value that drops the model score below the decision threshold.

This is a single-feature approach (changes one feature at a time), NOT joint
optimization. It answers: "If only X changed, what would X need to be for this
transaction to not be flagged?"

Limitations:
  - Single-feature counterfactuals may be infeasible if multiple features jointly
    push a score over threshold. A joint approach (e.g. DiCE) handles this.
  - The resulting values may be unrealistic for correlated features.
  - Sweeping toward zero is a reasonable heuristic for risk-increasing features
    but may not reflect the true "nearest" counterfactual in feature space.

This is intended as an operational decision aid ("what would need to change?"),
not as a causally valid counterfactual in the philosophical sense.
"""

import pandas as pd


def find_counterfactuals(
    model,
    X: pd.DataFrame,
    threshold: float,
    top_features: list[dict],
    n_steps: int = 100,
) -> list[dict]:
    """
    For each risk-increasing feature in top_features, sweep its value toward
    zero and find the minimum change that drops the model score below threshold.

    Args:
        model:        Fitted LightGBM classifier (or any sklearn-compatible model).
        X:            Single-row feature DataFrame used for scoring.
        threshold:    Decision threshold (score >= threshold -> flagged).
        top_features: Top-k SHAP features from shap_utils.top_k_features().
        n_steps:      Sweep resolution (higher = more precise counterfactual value).

    Returns:
        List of counterfactual dicts. Empty if not flagged or no single feature
        can flip the prediction alone.

        Each dict contains:
          - feature: str
          - current_value: float
          - counterfactual_value: float (value that drops score below threshold)
          - score_after: float (score at counterfactual value)
          - pct_change: float (% change from current to counterfactual)
    """
    current_score = float(model.predict_proba(X)[0, 1])
    if current_score < threshold:
        return []

    risk_drivers = [
        f for f in top_features
        if f["direction"] == "increases_risk" and f["feature"] in X.columns
    ]

    results = []
    for feat in risk_drivers:
        feat_name = feat["feature"]
        current_val = float(X[feat_name].iloc[0])
        target = 0.0  # sweep toward zero (lower = less suspicious for risk drivers)

        flipped = False
        for step in range(1, n_steps + 1):
            t = step / n_steps
            trial_val = current_val * (1.0 - t) + target * t
            X_trial = X.copy()
            X_trial[feat_name] = trial_val
            trial_score = float(model.predict_proba(X_trial)[0, 1])

            if trial_score < threshold:
                pct_change = (
                    (trial_val - current_val) / (abs(current_val) + 1e-9) * 100
                )
                results.append({
                    "feature": feat_name,
                    "current_value": round(current_val, 4),
                    "counterfactual_value": round(trial_val, 4),
                    "score_after": round(trial_score, 4),
                    "pct_change": round(pct_change, 1),
                })
                flipped = True
                break

        if not flipped and current_val != target:
            # Full sweep to zero didn't flip - record the best we got
            X_trial = X.copy()
            X_trial[feat_name] = target
            trial_score = float(model.predict_proba(X_trial)[0, 1])
            if trial_score < threshold:
                results.append({
                    "feature": feat_name,
                    "current_value": round(current_val, 4),
                    "counterfactual_value": round(target, 4),
                    "score_after": round(trial_score, 4),
                    "pct_change": round(-100.0, 1),
                })

    return results
