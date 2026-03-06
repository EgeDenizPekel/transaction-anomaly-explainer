"""
Faithfulness evaluation framework.

Measures whether an LLM-generated explanation faithfully reflects the SHAP
feature attribution used as ground truth.

Two eval contexts (per design decision):

  v1 (unconstrained prompt):
    - mention_rate:      fraction of top-3 SHAP features referenced in the explanation
    - direction_accuracy: fraction of mentioned features with correct risk direction

  v2 (constrained prompt):
    - mention_rate is trivially ~1.0 by prompt design - not reported as a signal
    - direction_accuracy: fraction of features with correct risk direction
    - value_accuracy:    fraction of features where the numeric value appears in explanation
    - hallucination_rate: fraction of explanations introducing feature names not in top-3

composite_faithfulness (reported for both):
    direction_accuracy * (mention_rate for v1, 1.0 for v2)
"""

import re
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synonym sets for direction detection
# ---------------------------------------------------------------------------
# For each direction, words/phrases that correctly express that direction.
# "increases_risk" features should be described as high/unusual/above-normal.
# "decreases_risk" features should be described as low/normal/below-normal.

INCREASES_RISK_TERMS = {
    "high", "higher", "large", "larger", "elevated", "above",
    "unusual", "unusually", "exceed", "exceeds", "exceeded",
    "suspicious", "abnormal", "outlier", "spike", "rapid",
    "multiple", "frequent", "new", "unfamiliar", "first",
    "velocity", "never", "previously unseen",
}

DECREASES_RISK_TERMS = {
    "low", "lower", "small", "smaller", "below", "normal",
    "typical", "within", "expected", "ordinary", "regular",
    "familiar", "known", "consistent", "average",
}


def _normalize(text: str) -> str:
    return text.lower()


def _feature_mentioned(explanation: str, feature_name: str) -> bool:
    """
    Check if a feature name (or a readable alias) is mentioned in the explanation.
    We match on the feature name itself and common human-readable variants.
    """
    text = _normalize(explanation)
    name = feature_name.lower()

    # Direct match
    if name in text:
        return True

    # Alias map for known features
    aliases = {
        "transactionamt": ["amount", "transaction amount", "amt"],
        "transactionamt_zscore": ["amount", "standard deviation", "z-score", "zscore", "unusual amount", "average"],
        "transactionamt_log": ["amount", "transaction amount"],
        "amt_to_mean_ratio": ["amount", "average", "ratio", "mean"],
        "card_amt_std": ["variability", "standard deviation", "volatility", "variation"],
        "card_amt_mean": ["average", "mean", "typical amount"],
        "txn_velocity_1h": ["velocity", "transactions", "frequency", "rapid", "multiple", "hour"],
        "time_since_last_txn": ["time", "interval", "gap", "previous", "last transaction", "recently"],
        "is_new_device": ["device", "new device", "unfamiliar", "first time", "never"],
        "has_identity": ["identity", "device", "verified"],
        "hour_of_day": ["hour", "time of day", "night", "morning", "evening"],
        "day_of_week": ["day", "weekend", "weekday"],
        "card1": ["card", "account"],
        "card2": ["card", "account"],
        "p_emaildomain": ["email", "domain"],
        "r_emaildomain": ["email", "domain"],
        "addr1": ["address", "location"],
        "addr2": ["address", "location"],
    }

    feature_key = name.replace("_", "").replace(" ", "").lower()
    for key, alias_list in aliases.items():
        if key == feature_key or key in name:
            if any(alias in text for alias in alias_list):
                return True

    return False


def _direction_correct(explanation: str, direction: str) -> Optional[bool]:
    """
    Check if the explanation correctly reflects the SHAP direction.
    Returns True/False/None (None = direction ambiguous in explanation).
    """
    text = _normalize(explanation)
    has_increase = any(term in text for term in INCREASES_RISK_TERMS)
    has_decrease = any(term in text for term in DECREASES_RISK_TERMS)

    if direction == "increases_risk":
        if has_increase and not has_decrease:
            return True
        if has_decrease and not has_increase:
            return False
    elif direction == "decreases_risk":
        if has_decrease and not has_increase:
            return True
        if has_increase and not has_decrease:
            return False

    # Both or neither - ambiguous
    return None


def _value_present(explanation: str, feature_value: float) -> bool:
    """Check if the numeric feature value (or a close rounding) appears in the explanation."""
    text = explanation

    # Extract all numbers from the explanation
    numbers_in_text = re.findall(r"\d+\.?\d*", text)
    numbers_in_text = [float(n) for n in numbers_in_text]

    # Match if within 5% or within 0.1 absolute
    for n in numbers_in_text:
        if abs(n - abs(feature_value)) <= max(0.1, abs(feature_value) * 0.05):
            return True
    return False


def _hallucination_check(explanation: str, top_features: list[dict]) -> bool:
    """
    Returns True if the explanation introduces a feature concept NOT present
    in any of the top features. Checks against a set of known feature domains
    and flags if a domain outside the top features is mentioned.
    """
    top_names = {f["feature"].lower() for f in top_features}

    # Domain groups - if ANY feature in a group is in top_features, that domain is fair game
    domain_groups = {
        "amount": {"transactionamt", "transactionamt_zscore", "transactionamt_log",
                   "amt_to_mean_ratio", "card_amt_std", "card_amt_mean"},
        "velocity": {"txn_velocity_1h", "time_since_last_txn"},
        "device": {"is_new_device", "has_identity", "deviceinfo", "devicetype"},
        "time": {"hour_of_day", "day_of_week"},
        "card": {"card1", "card2", "card3", "card4", "card5", "card6"},
        "email": {"p_emaildomain", "r_emaildomain"},
        "address": {"addr1", "addr2"},
    }

    allowed_domains = set()
    for domain, members in domain_groups.items():
        if top_names & members:
            allowed_domains.add(domain)

    # Terms that would indicate referencing a domain NOT in top features
    domain_terms = {
        "amount":   ["amount", "transaction amount", "dollar", "$", "price"],
        "velocity": ["velocity", "frequency", "multiple transaction", "rapid", "burst"],
        "device":   ["device", "new device", "unfamiliar device"],
        "time":     ["hour", "time of day", "night", "morning"],
        "card":     ["card number", "card type", "card network"],
        "email":    ["email", "email domain"],
        "address":  ["address", "location", "zip"],
    }

    text = _normalize(explanation)
    for domain, terms in domain_terms.items():
        if domain not in allowed_domains:
            if any(term in text for term in terms):
                return True  # hallucination detected

    return False


# ---------------------------------------------------------------------------
# Per-transaction faithfulness
# ---------------------------------------------------------------------------

def compute_faithfulness(
    explanation: str,
    top_features: list[dict],
    prompt_version: str = "v2",
) -> dict:
    """
    Compute faithfulness metrics for a single (explanation, shap_features) pair.

    Args:
        explanation:    LLM-generated explanation string
        top_features:   List of dicts from shap_utils.top_k_features() (top 3)
        prompt_version: "v1" or "v2" - determines which metrics are meaningful

    Returns dict with all metrics (consumer decides which to report per version).
    """
    features = top_features[:3]

    mention_results = [_feature_mentioned(explanation, f["feature"]) for f in features]
    mention_rate = sum(mention_results) / len(features)

    direction_results = []
    for f, mentioned in zip(features, mention_results):
        if mentioned:
            correct = _direction_correct(explanation, f["direction"])
            if correct is not None:
                direction_results.append(correct)

    direction_accuracy = (
        sum(direction_results) / len(direction_results) if direction_results else None
    )

    value_results = [
        _value_present(explanation, f["feature_value"])
        for f in features
        if f["feature_value"] not in (-999.0, 0.0)  # skip sentinel/zero values
    ]
    value_accuracy = sum(value_results) / len(value_results) if value_results else None

    hallucinated = _hallucination_check(explanation, features)

    # Composite: for v1 weight by mention_rate; for v2 assume mention=1.0
    da = direction_accuracy if direction_accuracy is not None else 0.0
    composite = mention_rate * da if prompt_version == "v1" else da

    return {
        "mention_rate": round(mention_rate, 4),
        "direction_accuracy": round(direction_accuracy, 4) if direction_accuracy is not None else None,
        "value_accuracy": round(value_accuracy, 4) if value_accuracy is not None else None,
        "hallucinated": hallucinated,
        "composite_faithfulness": round(composite, 4),
        "prompt_version": prompt_version,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    records: list[dict],
    prompt_version: str = "v2",
) -> dict:
    """
    Aggregate faithfulness metrics across a batch of records.

    Each record: {"explanation": str, "top_features": list[dict]}

    Returns summary statistics dict.
    """
    results = [
        compute_faithfulness(r["explanation"], r["top_features"], prompt_version)
        for r in records
    ]

    mention_rates      = [r["mention_rate"] for r in results]
    direction_accs     = [r["direction_accuracy"] for r in results if r["direction_accuracy"] is not None]
    value_accs         = [r["value_accuracy"] for r in results if r["value_accuracy"] is not None]
    composites         = [r["composite_faithfulness"] for r in results]
    hallucination_rate = sum(r["hallucinated"] for r in results) / len(results)

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "n": len(results),
        "prompt_version": prompt_version,
        "mean_mention_rate": safe_mean(mention_rates),
        "mean_direction_accuracy": safe_mean(direction_accs),
        "mean_value_accuracy": safe_mean(value_accs),
        "mean_composite_faithfulness": safe_mean(composites),
        "hallucination_rate": round(hallucination_rate, 4),
        "individual": results,
    }
