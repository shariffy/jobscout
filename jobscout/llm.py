"""Shared OpenRouter access.

Every LLM call in jobscout — bulk scoring, profile extraction, and prep briefs —
goes through OpenRouter's OpenAI-compatible surface, so the whole app needs only a
single OPENROUTER_API_KEY and any model slug OpenRouter serves works everywhere.
"""
from __future__ import annotations

import time

import openai

from .config import Config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_client(cfg: Config) -> openai.OpenAI:
    """Build an OpenAI client pointed at OpenRouter, using the configured key."""
    key = cfg.ai.openrouter_api_key
    if not key:
        raise RuntimeError(
            "No OpenRouter API key — set OPEN_ROUTER_API_KEY (or OPENROUTER_API_KEY) "
            "in the environment, or ai.openrouter_api_key in config.toml."
        )
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def with_retries(fn, label: str, max_retries: int = 5):
    """Retry on rate limits (429) with exponential backoff, honouring Retry-After.
    Non-rate-limit errors are raised immediately for the caller to handle."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            msg = str(exc).lower()
            is_rate = status == 429 or "429" in msg or ("rate" in msg and "limit" in msg)
            if not (is_rate and attempt < max_retries):
                raise
            retry_after = None
            resp = getattr(exc, "response", None)
            if resp is not None and hasattr(resp, "headers"):
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                try:
                    retry_after = float(ra) if ra else None
                except (TypeError, ValueError):
                    retry_after = None
            wait = retry_after if retry_after is not None else delay
            time.sleep(wait)
            delay = min(delay * 2, 30)
