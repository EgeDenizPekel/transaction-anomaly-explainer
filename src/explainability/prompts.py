"""
Prompt templates for the LLM explanation layer.

Two versions:
- v1: Unconstrained - asks for an explanation with no feature grounding
- v2: Constrained  - explicitly lists SHAP top-3 features and instructs the model
                     to reference each one and nothing else

The faithfulness experiment compares these two versions.
"""

SYSTEM_PROMPT = (
    "You are a fraud analyst assistant. You write concise, factual explanations of "
    "why a transaction was flagged as potentially fraudulent. "
    "Do not add context not present in the data you are given. "
    "Do not speculate about intent. Write 2-3 sentences only."
)


def build_prompt_v1(score: float, features: list[dict]) -> str:
    """
    Unconstrained prompt. Provides the anomaly score and raw feature values
    but gives no explicit instruction to reference specific features.
    """
    feature_lines = "\n".join(
        f"  - {f['feature']}: {f['feature_value']}" for f in features
    )
    return (
        f"A transaction has been flagged with an anomaly score of {score:.2f} "
        f"(higher = more suspicious).\n\n"
        f"Transaction details:\n{feature_lines}\n\n"
        f"Write a 2-3 sentence explanation for a compliance analyst describing "
        f"why this transaction was flagged."
    )


def build_prompt_v2(score: float, features: list[dict]) -> str:
    """
    Constrained prompt. Explicitly names the top-3 SHAP features and their
    values, and instructs the model to reference each one specifically.
    This is the mechanism that improves faithfulness.
    """
    assert len(features) >= 3, "v2 prompt requires at least 3 features"

    def direction_word(f: dict) -> str:
        return "increases fraud risk" if f["direction"] == "increases_risk" else "decreases fraud risk"

    lines = "\n".join(
        f"{i+1}. {f['feature']}: value={f['feature_value']}, "
        f"impact={f['shap_value']:+.3f} ({direction_word(f)})"
        for i, f in enumerate(features[:3])
    )

    return (
        f"A transaction has been flagged with anomaly score {score:.2f}.\n\n"
        f"The model's top reasons (by importance):\n{lines}\n\n"
        f"Write a 2-3 sentence explanation for a compliance analyst. "
        f"You MUST reference each of the three factors listed above by name. "
        f"Be specific about the values. "
        f"Do not mention any features or reasons not listed above."
    )
