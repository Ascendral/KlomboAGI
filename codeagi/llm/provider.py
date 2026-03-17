"""LLM provider using only stdlib (urllib.request).

Supports OpenAI-compatible APIs (OpenAI, GPT, Ollama, Groq, DeepSeek) and
Anthropic Claude API.  Auto-detects format based on model name.

Configuration via environment variables:
  CODEAGI_LLM_BASE_URL  — API base URL
                           Claude default: https://api.anthropic.com
                           OpenAI default: https://api.openai.com/v1
  CODEAGI_LLM_MODEL     — Model name (default: claude-sonnet-4-20250514)
  CODEAGI_LLM_API_KEY   — API key (checks CODEAGI_LLM_API_KEY, then ANTHROPIC_API_KEY, then OPENAI_API_KEY)
  CODEAGI_LLM_ENABLED   — Set to "1" to enable LLM calls (default: "1")
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _model() -> str:
    return os.getenv("CODEAGI_LLM_MODEL", "claude-sonnet-4-20250514")


def _is_claude() -> bool:
    return _model().startswith("claude")


def _api_key() -> str:
    key = os.getenv("CODEAGI_LLM_API_KEY", "")
    if key:
        return key
    if _is_claude():
        return os.getenv("ANTHROPIC_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")


def _base_url() -> str:
    explicit = os.getenv("CODEAGI_LLM_BASE_URL", "")
    if explicit:
        return explicit.rstrip("/")
    if _is_claude():
        return "https://api.anthropic.com"
    return "https://api.openai.com/v1"


def _enabled() -> bool:
    return os.getenv("CODEAGI_LLM_ENABLED", "1") == "1"


def _complete_claude(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Send a message to the Anthropic Claude API."""
    url = f"{_base_url()}/v1/messages"
    payload = {
        "model": _model(),
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _api_key(),
        "anthropic-version": "2023-06-01",
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data.get("content", [])
    if not content:
        return ""
    return str(content[0].get("text", "")).strip()


def _complete_openai(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Send a chat completion request to an OpenAI-compatible endpoint."""
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", "")).strip()


def complete(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Send a completion request, auto-detecting Claude vs OpenAI format.

    Returns the assistant message content, or an empty string if the LLM is
    unavailable, disabled, or returns an error — allowing callers to fall back
    to existing heuristics.
    """
    if not _enabled():
        return ""

    try:
        if _is_claude():
            return _complete_claude(system_prompt, user_prompt, temperature)
        return _complete_openai(system_prompt, user_prompt, temperature)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError) as exc:
        import sys
        print(f"[CodeAGI LLM] {type(exc).__name__}: {exc}", file=sys.stderr)
        return ""
