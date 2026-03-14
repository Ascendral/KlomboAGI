"""Minimal LLM provider using only stdlib (urllib.request).

Supports any OpenAI-compatible API: OpenAI, Ollama, Groq, DeepSeek, etc.
Configuration via environment variables:
  CODEAGI_LLM_BASE_URL  — API base URL (default: http://localhost:11434/v1)
  CODEAGI_LLM_MODEL     — Model name   (default: qwen3:14b)
  CODEAGI_LLM_API_KEY   — API key      (default: empty string, for Ollama)
  CODEAGI_LLM_ENABLED   — Set to "1" to enable LLM calls (default: "0")
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _base_url() -> str:
    return os.getenv("CODEAGI_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")


def _model() -> str:
    return os.getenv("CODEAGI_LLM_MODEL", "qwen3:14b")


def _api_key() -> str:
    return os.getenv("CODEAGI_LLM_API_KEY", "")


def _enabled() -> bool:
    return os.getenv("CODEAGI_LLM_ENABLED", "0") == "1"


def complete(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Send a chat completion request to an OpenAI-compatible endpoint.

    Returns the assistant message content, or an empty string if the LLM is
    unavailable, disabled, or returns an error — allowing callers to fall back
    to existing heuristics.

    LLM calls are disabled by default.  Set CODEAGI_LLM_ENABLED=1 to activate.
    """
    if not _enabled():
        return ""

    url = f"{_base_url()}/chat/completions"
    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = _api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError):
        return ""
