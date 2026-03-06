"""
LiteLLM wrapper for generating natural-language fraud explanations.

Supports:
- Ollama (local dev, no API cost): LLM_PROVIDER=ollama
- OpenAI (production):             LLM_PROVIDER=openai

Switch providers via environment variable with zero application code changes.
"""

import os
import logging
from typing import Optional

import litellm

from src.explainability.prompts import SYSTEM_PROMPT, build_prompt_v1, build_prompt_v2, build_prompt_v3

log = logging.getLogger(__name__)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

PROVIDER_CONFIG = {
    "ollama": {
        "model": os.getenv("OLLAMA_MODEL", "ollama/llama3.1:8b"),
        "api_base": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": 0.1,
        "max_tokens": 150,
    },
    "openai": {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.1,
        "max_tokens": 150,
    },
}


def _get_config() -> dict:
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER)
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose: {list(PROVIDER_CONFIG)}")
    return PROVIDER_CONFIG[provider]


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def _call_llm(user_prompt: str, config: dict, timeout: int, max_tokens: int | None = None) -> str:
    """Call the LLM with a given prompt. Returns raw response text."""
    response = litellm.completion(
        model=config["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config["temperature"],
        max_tokens=max_tokens or config["max_tokens"],
        timeout=timeout,
        api_base=config.get("api_base"),
    )
    return response.choices[0].message.content.strip()


def generate_explanation(
    score: float,
    features: list[dict],
    prompt_version: str = "v2",
    timeout: int = 60,
) -> str:
    """
    Generate a natural-language summary of model attribution signals for a
    flagged transaction.

    The explanation describes what the model observed in the input features.
    It does NOT claim the transaction is fraudulent - it describes model score
    drivers using SHAP attribution as the reference signal.

    Args:
        score:          Anomaly score (0-1).
        features:       Top-k SHAP features from shap_utils.top_k_features().
        prompt_version: "v1" (unconstrained), "v2" (constrained), or "v3" (structured JSON).
        timeout:        Request timeout in seconds.

    Returns:
        Plain-text summary string.
    """
    config = _get_config()

    try:
        if prompt_version == "v1":
            user_prompt = build_prompt_v1(score, features)
            return _call_llm(user_prompt, config, timeout)

        elif prompt_version == "v2":
            user_prompt = build_prompt_v2(score, features)
            return _call_llm(user_prompt, config, timeout)

        elif prompt_version == "v3":
            return _generate_v3(score, features, config, timeout)

        else:
            raise ValueError(f"Unknown prompt_version '{prompt_version}'. Choose: v1, v2, v3")

    except Exception as e:
        log.error(f"LLM generation failed: {e}")
        raise


def _generate_v3(score: float, features: list[dict], config: dict, timeout: int) -> str:
    """
    Structured generation: request JSON from LLM, validate structure, extract
    prose summary. Falls back to v2 if JSON parsing or validation fails.

    This separates model-grounded structure from prose generation - the JSON
    is validated for required fields before the summary is accepted.
    """
    import json

    user_prompt = build_prompt_v3(score, features)
    raw = _call_llm(user_prompt, config, timeout, max_tokens=300)

    # Try to parse JSON (LLM may wrap it in markdown code fences)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Strip opening and closing fences
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
        # Validate required structure
        if "summary" not in data or "primary_drivers" not in data:
            raise ValueError("Missing required keys in v3 JSON response")
        drivers = data["primary_drivers"]
        if not isinstance(drivers, list) or len(drivers) < 3:
            raise ValueError("primary_drivers must have at least 3 entries")
        for d in drivers:
            if "feature" not in d or "contribution" not in d:
                raise ValueError("Each driver must have 'feature' and 'contribution' keys")
        return str(data["summary"]).strip()

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        log.warning(f"v3 JSON parsing/validation failed: {e}. Falling back to v2.")
        fallback_prompt = build_prompt_v2(score, features)
        return _call_llm(fallback_prompt, config, timeout)
