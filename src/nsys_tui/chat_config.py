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
    # Anthropic (Claude 4.x)
    {"id": "anthropic/claude-opus-4-6-20260205",      "label": "Claude Opus 4.6"},
    {"id": "anthropic/claude-sonnet-4-6",              "label": "Claude Sonnet 4.6"},
    {"id": "anthropic/claude-sonnet-4-5-20250929",     "label": "Claude Sonnet 4.5"},
    {"id": "anthropic/claude-opus-4-5-20251101",       "label": "Claude Opus 4.5"},
    {"id": "anthropic/claude-haiku-4-5-20251001",      "label": "Claude Haiku 4.5"},
    # OpenAI (GPT-5.x)
    {"id": "gpt-5.2",                                  "label": "GPT-5.2"},
    {"id": "gpt-5.2-pro",                              "label": "GPT-5.2 Pro"},
    {"id": "gpt-5-mini",                               "label": "GPT-5 Mini"},
    {"id": "gpt-5.3-codex",                            "label": "GPT-5.3 Codex"},
    {"id": "gpt-4o",                                   "label": "GPT-4o"},
    # Gemini (2.5+ and 3.x)
    {"id": "gemini/gemini-2.5-pro",                    "label": "Gemini 2.5 Pro"},
    {"id": "gemini/gemini-2.5-pro-preview-05-20",      "label": "Gemini 2.5 Pro (preview)"},
    {"id": "gemini/gemini-2.5-flash",                  "label": "Gemini 2.5 Flash"},
    {"id": "gemini/gemini-2.5-flash-lite",             "label": "Gemini 2.5 Flash Lite"},
    {"id": "gemini/gemini-2.0-flash",                  "label": "Gemini 2.0 Flash"},
    {"id": "gemini/gemini-3.1-pro-preview",            "label": "Gemini 3.1 Pro"},
    {"id": "gemini/gemini-3-pro-preview",              "label": "Gemini 3 Pro"},
    {"id": "gemini/gemini-3-flash-preview",            "label": "Gemini 3 Flash"},
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
        return "anthropic/claude-sonnet-4-5-20250929", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-5.2", os.environ["OPENAI_API_KEY"]
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
