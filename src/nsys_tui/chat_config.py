"""
chat_config.py — Model registry and API-key resolution for the AI chat layer.

Keeping model discovery separate from the agent loop makes it easy to test
key resolution without triggering any LLM imports.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Supported models — ordered by preference within each provider.
# Model IDs must match the LiteLLM provider/model naming convention.
# Anthropic 4.x, OpenAI 5.x, Gemini 2.5+ and 3.x only.
# ---------------------------------------------------------------------------
MODEL_OPTIONS: list[dict] = [
    # Anthropic
    {"id": "anthropic/claude-sonnet-4-20250514",        "label": "Claude Sonnet 4"},
    {"id": "anthropic/claude-3.5-sonnet",               "label": "Claude 3.5 Sonnet"},
    {"id": "anthropic/claude-3-5-haiku-20241022",       "label": "Claude 3.5 Haiku"},
    # OpenAI
    {"id": "gpt-4o",                                    "label": "GPT-4o"},
    {"id": "gpt-4o-mini",                               "label": "GPT-4o Mini"},
    {"id": "o3-mini",                                   "label": "o3-mini"},
    # Gemini
    {"id": "gemini/gemini-2.5-flash",                   "label": "Gemini 2.5 Flash"},
    {"id": "gemini/gemini-2.5-pro",                     "label": "Gemini 2.5 Pro"},
    {"id": "gemini/gemini-2.0-flash",                   "label": "Gemini 2.0 Flash"},
]


def _model_to_key(model_id: str) -> str | None:
    """Return the environment-variable name that holds the API key for *model_id*.

    Returns ``None`` for unrecognised providers.
    """
    if not model_id:
        return None
    if model_id.startswith("anthropic/"):
        return "ANTHROPIC_API_KEY"
    if model_id.startswith("openai/") or model_id.startswith("gpt-"):
        return "OPENAI_API_KEY"
    if model_id.startswith("gemini/"):
        return "GEMINI_API_KEY"
    return None


def _get_model_and_key(
    preferred_model: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the model ID and API key to use for a completion request.

    Priority:
    1. *preferred_model* (from request payload or ``NSYS_AI_MODEL`` env var),
       if its provider's API key is set.
    2. First provider with a configured key: Anthropic → OpenAI → Gemini.

    Returns ``(model_id, api_key)`` or ``(None, None)`` when nothing is configured.
    """
    preferred = preferred_model or os.environ.get("NSYS_AI_MODEL")
    if preferred:
        key_name = _model_to_key(preferred)
        if key_name and os.environ.get(key_name):
            return preferred, os.environ[key_name]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-20250514", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o", os.environ["OPENAI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-pro", os.environ["GEMINI_API_KEY"]
    return None, None


def get_available_models() -> list[dict]:
    """Return ``[{id, label}, ...]`` for every model whose API key is present."""
    return [
        {"id": opt["id"], "label": opt["label"]}
        for opt in MODEL_OPTIONS
        if (key := _model_to_key(opt["id"])) and os.environ.get(key)
    ]


def get_default_model() -> str | None:
    """Return the model ID to use when no explicit model is requested."""
    if os.environ.get("NSYS_AI_MODEL"):
        key_name = _model_to_key(os.environ["NSYS_AI_MODEL"])
        if key_name and os.environ.get(key_name):
            return os.environ["NSYS_AI_MODEL"]
    models = get_available_models()
    return models[0]["id"] if models else None
