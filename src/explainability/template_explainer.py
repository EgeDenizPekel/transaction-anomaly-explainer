"""
Template-based explanation generator used as a non-LLM baseline.

Produces deterministic, rule-based natural language from SHAP top features
with no language model involvement. Used in faithfulness evaluation to measure
what, if anything, LLM explanations add over simple structured templates.

A good LLM explanation should:
  - Match or exceed template faithfulness (mention_rate, direction_accuracy)
  - Beat template on fluency and contextual framing
  - Not introduce hallucinations the template never would

A template baseline always has hallucination_rate = 0.0 by construction,
since it only references the features explicitly passed to it.
"""


def _format_value(feature: str, value: float) -> str:
    if value == -999.0:
        return "missing"
    if feature == "time_since_last_txn":
        if value < 0:
            return "N/A"
        if value < 60:
            return f"{value:.0f}s"
        if value < 3600:
            return f"{value / 60:.1f} min"
        if value < 86400:
            return f"{value / 3600:.1f} hrs"
        return f"{value / 86400:.1f} days"
    if feature == "txn_velocity_1h":
        return f"{value:.0f} txns/hr"
    if feature in ("TransactionAmt", "card_amt_mean", "card_amt_std"):
        return f"${value:.2f}"
    if feature == "TransactionAmt_zscore":
        return f"{value:.2f} sigma"
    if feature == "hour_of_day":
        return f"{value:.0f}:00"
    if feature == "is_new_device":
        return "yes" if value >= 0.5 else "no"
    return f"{value:.4f}"


_RISK_PHRASES = {
    "txn_velocity_1h": "elevated transaction velocity ({value})",
    "TransactionAmt_zscore": "unusually large transaction amount ({value} above card mean)",
    "hour_of_day": "transaction at an atypical hour ({value})",
    "is_new_device": "use of a previously unseen device",
    "time_since_last_txn": "unusually short interval since last transaction ({value})",
    "TransactionAmt": "large transaction amount ({value})",
    "card_amt_std": "high variability in card transaction amounts ({value})",
    "_default": "elevated value for {feature} ({value})",
}

_REDUCE_PHRASES = {
    "txn_velocity_1h": "low transaction velocity ({value})",
    "TransactionAmt_zscore": "transaction amount within normal range ({value})",
    "hour_of_day": "transaction during typical hours ({value})",
    "is_new_device": "use of a recognized device",
    "time_since_last_txn": "typical interval since last transaction ({value})",
    "TransactionAmt": "transaction amount within normal range ({value})",
    "_default": "{feature} within expected range ({value})",
}


def _phrase_for(feature: str, value: float, direction: str) -> str:
    phrases = _RISK_PHRASES if direction == "increases_risk" else _REDUCE_PHRASES
    template = phrases.get(feature, phrases["_default"])
    return template.format(feature=feature, value=_format_value(feature, value))


def generate_template_explanation(score: float, features: list[dict]) -> str:
    """
    Generate a deterministic template explanation from top-3 SHAP features.

    Args:
        score:    Anomaly score (0-1).
        features: List of top-k feature dicts from shap_utils.top_k_features().

    Returns:
        Plain-text explanation string. No LLM involved.
    """
    features = features[:3]
    risk_drivers = [f for f in features if f["direction"] == "increases_risk"]
    reducing = [f for f in features if f["direction"] == "decreases_risk"]
    score_pct = f"{score * 100:.1f}%"

    if not risk_drivers:
        return (
            f"The model assigned a risk score of {score_pct}. "
            f"No strong risk-increasing attribution signals were identified among the top features."
        )

    driver_phrases = [
        _phrase_for(f["feature"], f["feature_value"], f["direction"])
        for f in risk_drivers
    ]

    if len(driver_phrases) == 1:
        drivers_text = driver_phrases[0]
    elif len(driver_phrases) == 2:
        drivers_text = f"{driver_phrases[0]} and {driver_phrases[1]}"
    else:
        drivers_text = f"{driver_phrases[0]}, {driver_phrases[1]}, and {driver_phrases[2]}"

    sentence1 = (
        f"The model assigned a risk score of {score_pct}, "
        f"driven by {drivers_text}."
    )

    if reducing:
        reducing_phrase = _phrase_for(
            reducing[0]["feature"], reducing[0]["feature_value"], reducing[0]["direction"]
        )
        sentence2 = f"Partially offsetting: {reducing_phrase}."
        return f"{sentence1} {sentence2}"

    return sentence1
