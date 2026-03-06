"""
Prompt templates for the LLM explanation layer.

Three versions:
- v1: Unconstrained - asks for a summary with no feature grounding constraint.
- v2: Constrained   - explicitly lists top-3 SHAP features and instructs the
                      model to reference each one and nothing else.
- v3: Structured    - asks for a JSON object (drivers + summary), which is
                      validated before the prose is extracted. Falls back to v2
                      if JSON parsing fails.

The faithfulness experiment compares v1 vs v2 vs a template baseline.

Language note: All prompts use "risk score" and "attribution signal" language,
not "this transaction is fraudulent". The model provides a prediction score,
not a determination of fraud. Maintaining this distinction prevents the LLM
from smuggling model behavior into claims about reality.
"""

SYSTEM_PROMPT = (
    "You are a fraud risk analyst assistant. You write concise, factual summaries "
    "of which model attribution signals drove an elevated risk score for a transaction. "
    "Describe what the model observed in the input features. "
    "Do not claim the transaction is fraudulent or assign intent to the event. "
    "Do not add context not present in the data you are given. "
    "Write 2-3 sentences only."
)


def build_prompt_v1(score: float, features: list[dict]) -> str:
    """
    Unconstrained prompt. Provides the anomaly score and raw feature values
    but gives no explicit instruction to reference specific features.
    Baseline for faithfulness evaluation.
    """
    feature_lines = "\n".join(
        f"  - {f['feature']}: {f['feature_value']}" for f in features
    )
    return (
        f"A transaction received an anomaly score of {score:.2f} "
        f"(scale 0-1, higher = higher modeled risk).\n\n"
        f"Model input features:\n{feature_lines}\n\n"
        f"Write a 2-3 sentence summary for a compliance analyst describing "
        f"which input features contributed most to this elevated risk score."
    )


def build_prompt_v2(score: float, features: list[dict]) -> str:
    """
    Constrained prompt. Explicitly names the top-3 SHAP attribution signals and
    their values, and instructs the model to reference each one specifically.
    This constraint is the mechanism that improves faithfulness vs v1.
    """
    assert len(features) >= 3, "v2 prompt requires at least 3 features"

    def direction_word(f: dict) -> str:
        return "increases risk score" if f["direction"] == "increases_risk" else "decreases risk score"

    lines = "\n".join(
        f"{i+1}. {f['feature']}: value={f['feature_value']}, "
        f"impact={f['shap_value']:+.3f} ({direction_word(f)})"
        for i, f in enumerate(features[:3])
    )

    return (
        f"A transaction received an anomaly score of {score:.2f}.\n\n"
        f"The model's top attribution signals (SHAP, by importance):\n{lines}\n\n"
        f"Write a 2-3 sentence summary for a compliance analyst. "
        f"You MUST reference each of the three signals listed above by name. "
        f"Be specific about the observed values. "
        f"Do not mention any features or reasons not listed above. "
        f"Do not claim the transaction is fraudulent - describe what the model observed."
    )


def build_prompt_v3(score: float, features: list[dict]) -> str:
    """
    Structured generation prompt. Asks the LLM to return a JSON object with
    structured driver data and a prose summary. The caller validates the JSON
    and extracts the 'summary' field, falling back to v2 if parsing fails.

    This separates model-grounded statements from narrative generation:
    the structure is validated before the prose is accepted.
    """
    assert len(features) >= 3, "v3 prompt requires at least 3 features"

    def direction_word(f: dict) -> str:
        return "increases risk score" if f["direction"] == "increases_risk" else "decreases risk score"

    lines = "\n".join(
        f"{i+1}. {f['feature']}: value={f['feature_value']}, "
        f"impact={f['shap_value']:+.3f} ({direction_word(f)})"
        for i, f in enumerate(features[:3])
    )

    return (
        f"A transaction received an anomaly score of {score:.2f}.\n\n"
        f"The model's top attribution signals (SHAP, by importance):\n{lines}\n\n"
        f"Return a JSON object with exactly this structure:\n"
        f'{{\n'
        f'  "primary_drivers": [\n'
        f'    {{"feature": "<name>", "observed": "<value description>", '
        f'"contribution": "increases|decreases risk score"}}\n'
        f'  ],\n'
        f'  "summary": "<2-3 sentence summary for a compliance analyst>"\n'
        f'}}\n\n'
        f"Requirements:\n"
        f"- primary_drivers must list all three signals above\n"
        f"- summary must reference each feature by name with specific values\n"
        f"- Do not claim the transaction is fraudulent - describe what the model observed\n"
        f"- Do not mention any features not in the list above\n"
        f"- Return ONLY the JSON object, no other text"
    )
