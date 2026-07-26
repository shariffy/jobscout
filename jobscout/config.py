from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .gates import GateConfig, PriorityConfig


class ProfileConfig(BaseModel):
    cv_path: str = "cv.pdf"
    goals: str = ""


class AIConfig(BaseModel):
    # Per-provider API key overrides, e.g. { openrouter = "...", google = "..." }.
    # Optional — each provider normally resolves its key from its own env var(s)
    # (see jobscout/llm.py PROVIDERS); set an entry here only to override that.
    api_keys: dict[str, str] = Field(default_factory=dict)
    # All three are "provider:model" strings, e.g. "google:gemini-3.1-flash-lite",
    # "openrouter:anthropic/claude-sonnet-4-6". No prefix defaults to openrouter, so
    # bare slugs from older configs keep working. Use a cheap route for bulk scoring
    # (runs on every listing) and stronger ones for deep/prep.
    bulk_model: str = "google:gemini-3.1-flash-lite"
    deep_model: str = "openrouter:anthropic/claude-sonnet-4-6"
    prep_model: str = "openrouter:anthropic/claude-sonnet-4-6"
    # Optional reasoning effort for the bulk model: "low"/"medium"/"high"/"max"/
    # "min"/"none". None = provider default. Each provider folds this into its own
    # request shape (see jobscout/llm.py); providers with no reasoning support
    # ignore it. Ignored entirely by providers with reasoning_style=None.
    bulk_reasoning_effort: str | None = None
    # Self-consistency: extract this many times per listing and take the majority
    # decision. >1 trades cost/latency for decision stability — the lever that lets
    # a cheaper, less-deterministic bulk_model match a stronger one.
    scorer_repeats: int = 1
    # How many listings to score concurrently in a batch (scan / rescore). Each
    # listing's vote already fans its mandatory repeats out in parallel, so the peak
    # in-flight request count is roughly scorer_concurrency * (scorer_repeats // 2 + 1)
    # — keep it modest so a shared provider key isn't rate-limited. 1 = serial.
    scorer_concurrency: int = 4


class NotionConfig(BaseModel):
    token: str = Field(default="")
    database_id: str = ""

    @model_validator(mode="after")
    def resolve_token(self) -> NotionConfig:
        if not self.token:
            self.token = os.environ.get("NOTION_TOKEN", "")
        return self


class StoreConfig(BaseModel):
    db_path: str = "jobscout.db"


class SelectorConfig(BaseModel):
    container: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    description: str = ""
    fetch_detail: bool = False  # fetch individual job page to enrich description


class SourceConfig(BaseModel):
    name: str
    type: Literal["http", "feed", "browser"]
    url: str
    enabled: bool = True
    selectors: SelectorConfig = Field(default_factory=SelectorConfig)
    login_url: str = ""
    username_env: str = ""
    password_env: str = ""
    # Pagination (http sources). Fetch `pages` pages by advancing the query
    # param `page_param` by `page_size` each time, starting from its current
    # value in the url. LinkedIn: page_param="start", page_size=10. Page-number
    # boards: page_param="page", page_size=1. Fetching stops early on an empty page.
    pages: int = 1
    page_param: str = ""
    page_size: int = 10


class Config(BaseModel):
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    gates: list[GateConfig] = Field(default_factory=list)
    priority: PriorityConfig = Field(default_factory=PriorityConfig)
    sources: list[SourceConfig] = Field(default_factory=list)


def assessment_config_hash(cfg: Config) -> str:
    """Fingerprint of everything that determines a score's output.

    Changing any of these makes prior scores stale: rescore --all uses this to skip
    listings already scored under the current config instead of re-paying for them.
    """
    payload = {
        "bulk_model": cfg.ai.bulk_model,
        "bulk_reasoning_effort": cfg.ai.bulk_reasoning_effort,
        "scorer_repeats": cfg.ai.scorer_repeats,
        "gates": [g.model_dump() for g in cfg.gates],
        "priority": cfg.priority.model_dump(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return digest[:16]


_CONFIG_PATHS = ["config.toml", "~/.config/jobscout/config.toml"]


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        for candidate in _CONFIG_PATHS:
            resolved = Path(candidate).expanduser()
            if resolved.exists():
                path = resolved
                break

    if path is None:
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    _warn_if_insecure(Path(path), raw)

    return Config.model_validate(raw)


def _warn_if_insecure(path: Path, raw: dict) -> None:
    """Best-effort nudge toward env vars: warn (don't fail) if config.toml is
    readable by other local users, or carries an inline secret that env vars
    would keep off disk."""
    if os.name == "posix":
        try:
            mode = path.stat().st_mode
        except OSError:
            mode = 0
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(
                f"[jobscout] warning: {path} is readable by group/other — "
                f"run `chmod 600 {path}` or keep secrets in env vars instead.",
                file=sys.stderr,
            )
    ai = raw.get("ai") or {}
    if isinstance(ai, dict) and any((ai.get("api_keys") or {}).values()):
        print(
            f"[jobscout] warning: {path} has an inline [ai.api_keys] secret — "
            "prefer the provider's env var (see jobscout/llm.py PROVIDERS) instead.",
            file=sys.stderr,
        )
    notion = raw.get("notion") or {}
    if isinstance(notion, dict) and notion.get("token"):
        print(
            f"[jobscout] warning: {path} has an inline notion.token — "
            "prefer the NOTION_TOKEN env var instead.",
            file=sys.stderr,
        )
