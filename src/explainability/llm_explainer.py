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

from src.explainability.prompts import SYSTEM_PROMPT, build_prompt_v1, build_prompt_v2

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

def generate_explanation(
    score: float,
    features: list[dict],
    prompt_version: str = "v2",
    timeout: int = 60,
) -> str:
    """
    Generate a natural-language explanation for a flagged transaction.

    Args:
        score:          Anomaly score (0-1)
        features:       Top-k SHAP features from shap_utils.top_k_features()
        prompt_version: "v1" (unconstrained) or "v2" (constrained)
        timeout:        Request timeout in seconds

    Returns:
        Plain-text explanation string.
    """
    if prompt_version == "v1":
        user_prompt = build_prompt_v1(score, features)
    elif prompt_version == "v2":
        user_prompt = build_prompt_v2(score, features)
    else:
        raise ValueError(f"Unknown prompt_version '{prompt_version}'. Choose: v1, v2")

    config = _get_config()

    try:
        response = litellm.completion(
            model=config["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=timeout,
            api_base=config.get("api_base"),
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        log.error(f"LLM generation failed: {e}")
        raise
