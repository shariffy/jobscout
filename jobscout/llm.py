"""Per-provider LLM routing.

Every model slot in config (ai.bulk_model / deep_model / prep_model) is a
"provider:model" string, e.g. "google:gemini-3.1-flash-lite" or
"openrouter:anthropic/claude-sonnet-4-6". A bare slug with no prefix defaults to
openrouter, so existing configs keep working unchanged. All four providers here
expose an OpenAI-compatible chat.completions surface, so one openai.OpenAI client
(per provider, cached) covers everything — gateways (openrouter, requesty) and
direct provider APIs (google, anthropic) alike.

Gateways and direct providers diverge on a few request/response details, so each
Provider entry also carries how to fold in reasoning effort, which server-side
web-search tool (if any) it offers, and whether it returns a real charged cost in
usage. Direct providers return token counts only — no cost field.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import openai

from .config import Config


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_envs: tuple[str, ...]
    # How a reasoning-effort string gets folded into extra_body:
    #   "openrouter-dict" -> {"reasoning": {"effort": e}}
    #   "reasoning_effort" -> {"reasoning_effort": e}
    #   None -> reasoning effort isn't supported; ignored with no error
    reasoning_style: str | None
    # Extra request fields always sent to this provider (e.g. asking a gateway to
    # include real charged cost in the response's usage object).
    base_extra_body: dict = field(default_factory=dict)
    # Extra_body fields that turn on server-side web search, merged wholesale into
    # the call's extra_body when a caller wants grounding; None if the provider
    # doesn't offer it. Shape varies per provider — OpenRouter/Requesty hang a tool
    # off the standard `tools` array, Google nests its own under a `google` key
    # (its grounding tool isn't part of the OpenAI-compatible tool-calling surface),
    # so this can't be a single normalized "tool" value.
    web_search_extra_body: dict | None = None


PROVIDERS: dict[str, Provider] = {
    "openrouter": Provider(
        base_url="https://openrouter.ai/api/v1",
        key_envs=("OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY"),
        reasoning_style="openrouter-dict",
        base_extra_body={"usage": {"include": True}},
        web_search_extra_body={"tools": [{"type": "openrouter:web_search"}]},
    ),
    "requesty": Provider(
        base_url="https://router.requesty.ai/v1",
        key_envs=("REQUESTY_API_KEY",),
        reasoning_style="reasoning_effort",
        web_search_extra_body={"tools": [{"type": "web_search_preview"}]},
    ),
    "google": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_envs=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        reasoning_style="reasoning_effort",
        # No web_search_extra_body: Grounding with Google Search is NOT available
        # through Gemini's OpenAI-compatibility endpoint — confirmed by a Google
        # staff reply on the Gemini API forum ("Grounding with Google search is
        # not available when using OpenAI compatibility mode"), and independently
        # reproduced here as a live 400 on every extra_body shape tried. Getting
        # grounding on Gemini needs the native Gemini SDK — a different,
        # non-OpenAI-compatible client this registry doesn't support. Don't
        # re-attempt this via extra_body; it isn't a syntax problem.
    ),
    "anthropic": Provider(
        base_url="https://api.anthropic.com/v1/",
        key_envs=("ANTHROPIC_API_KEY",),
        reasoning_style=None,
    ),
}

DEFAULT_PROVIDER = "openrouter"


def parse_model(spec: str) -> tuple[str, str]:
    """Split "provider:model" into (provider, model). No prefix -> openrouter,
    so bare slugs from existing configs keep resolving the way they always have."""
    if ":" in spec:
        provider, model = spec.split(":", 1)
        if provider in PROVIDERS:
            return provider, model
    return DEFAULT_PROVIDER, spec


_client_cache: dict[str, openai.OpenAI] = {}


def _provider_client(cfg: Config, provider_name: str) -> openai.OpenAI:
    if provider_name not in PROVIDERS:
        raise RuntimeError(f"Unknown provider {provider_name!r} — known: {', '.join(PROVIDERS)}")
    if provider_name in _client_cache:
        return _client_cache[provider_name]

    provider = PROVIDERS[provider_name]
    key = cfg.ai.api_keys.get(provider_name, "")
    if not key:
        for env in provider.key_envs:
            key = os.environ.get(env, "")
            if key:
                break
    if not key:
        envs = " or ".join(provider.key_envs)
        raise RuntimeError(
            f"No API key for provider {provider_name!r} — set {envs} in the environment, "
            f"or ai.api_keys.{provider_name} in config.toml."
        )
    client = openai.OpenAI(base_url=provider.base_url, api_key=key)
    _client_cache[provider_name] = client
    return client


@dataclass(frozen=True)
class Route:
    client: openai.OpenAI
    model: str
    provider_name: str
    provider: Provider


def resolve_route(cfg: Config, model_spec: str) -> Route:
    """Resolve a "provider:model" config string to a ready client + provider info."""
    provider_name, model = parse_model(model_spec)
    client = _provider_client(cfg, provider_name)
    return Route(
        client=client, model=model, provider_name=provider_name, provider=PROVIDERS[provider_name]
    )


def extra_body_for(provider: Provider, reasoning_effort: str | None) -> dict:
    """Build the extra_body for a call to this provider, folding in reasoning
    effort per the provider's own shape. Providers with no reasoning support
    (reasoning_style=None) silently ignore a configured effort rather than erroring,
    since not every model slot needs thinking control."""
    extra_body = dict(provider.base_extra_body)
    if reasoning_effort is not None:
        if provider.reasoning_style == "openrouter-dict":
            extra_body["reasoning"] = {"effort": reasoning_effort}
        elif provider.reasoning_style == "reasoning_effort":
            extra_body["reasoning_effort"] = reasoning_effort
    return extra_body


def with_retries(fn, label: str, max_retries: int = 5):
    """Retry on rate limits (429) with exponential backoff, honouring Retry-After.
    Non-rate-limit errors are raised immediately for the caller to handle.

    Logs each attempt's raw call duration to stderr, plus (on a retried call) the
    total wall time including backoff sleeps — so it's visible how much of any
    perceived slowness is provider latency vs. time spent waiting out 429s.
    """
    delay = 2.0
    start = time.monotonic()
    for attempt in range(max_retries + 1):
        call_start = time.monotonic()
        try:
            result = fn()
            call_elapsed = time.monotonic() - call_start
            if attempt == 0:
                print(f"[llm] {label}: {call_elapsed:.2f}s", file=sys.stderr)
            else:
                total_elapsed = time.monotonic() - start
                print(
                    f"[llm] {label}: {call_elapsed:.2f}s "
                    f"(succeeded after {attempt} retr{'y' if attempt == 1 else 'ies'}, "
                    f"total {total_elapsed:.2f}s incl. backoff)",
                    file=sys.stderr,
                )
            return result
        except Exception as exc:
            call_elapsed = time.monotonic() - call_start
            status = getattr(exc, "status_code", None)
            msg = str(exc).lower()
            is_rate = status == 429 or "429" in msg or ("rate" in msg and "limit" in msg)
            if not (is_rate and attempt < max_retries):
                print(
                    f"[llm] {label}: failed after {call_elapsed:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries + 1}) — {exc}",
                    file=sys.stderr,
                )
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
            print(
                f"[llm] {label}: 429 after {call_elapsed:.2f}s "
                f"(attempt {attempt + 1}/{max_retries + 1}), retrying in {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay = min(delay * 2, 30)
