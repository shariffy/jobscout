from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .gates import GateConfig, PriorityConfig


class ProfileConfig(BaseModel):
    cv_path: str = "cv.pdf"
    goals: str = ""


class AIConfig(BaseModel):
    # Every model call (bulk scoring, profile extraction, prep briefs) goes through
    # OpenRouter, so a single key powers the whole app. It resolves from
    # ai.openrouter_api_key or the OPEN_ROUTER_API_KEY / OPENROUTER_API_KEY env vars.
    openrouter_api_key: str = Field(default="")
    # All three are OpenRouter model slugs (provider/model[:variant]), e.g.
    # "anthropic/claude-haiku-4-5", "google/gemini-3.5-flash",
    # "qwen/qwen3-next-80b-a3b-instruct:free". Use a cheap one for bulk scoring
    # (runs on every listing) and stronger ones for deep/prep.
    bulk_model: str = "anthropic/claude-haiku-4-5"
    deep_model: str = "anthropic/claude-sonnet-4-6"
    prep_model: str = "anthropic/claude-sonnet-4-6"
    # Optional OpenRouter reasoning config for the bulk model, passed through
    # verbatim, e.g. {"effort": "low"} or {"enabled": false}. None = provider default.
    bulk_reasoning: dict[str, Any] | None = None
    # Which producer scores listings: "additive" = single 0-100 fit judged by the
    # LLM (legacy); "gated" = LLM extracts facts, Python decides apply/no + priority
    # from [[gates]] and [priority] (fit_score becomes a derived compat number).
    scorer: Literal["additive", "gated"] = "additive"
    # Self-consistency for the gated scorer: extract this many times per listing and
    # take the majority decision. >1 trades cost/latency for decision stability — the
    # lever that lets a cheaper, less-deterministic bulk_model match a stronger one.
    scorer_repeats: int = 1
    fit_threshold: int = 70

    @model_validator(mode="after")
    def resolve_api_key(self) -> AIConfig:
        if not self.openrouter_api_key:
            self.openrouter_api_key = (
                os.environ.get("OPEN_ROUTER_API_KEY", "")
                or os.environ.get("OPENROUTER_API_KEY", "")
            )
        return self


class NotionConfig(BaseModel):
    token: str = Field(default="")
    database_id: str = ""

    @model_validator(mode="after")
    def resolve_token(self) -> NotionConfig:
        if not self.token:
            self.token = os.environ.get("NOTION_TOKEN", "")
        return self


class ScoringWeights(BaseModel):
    """Maximum magnitude (points) each dimension can contribute to the score.

    Positive dimensions add up to this many points; negative dimensions
    (domain, salary, office, dealbreakers) subtract up to this many.
    """

    title: int = 20
    scope: int = 20
    company: int = 15
    stack: int = 10  # how well required skills/tools match the candidate
    domain: int = 10
    salary: int = 3
    office: int = 3
    dealbreakers: int = 50


class ScoringConfig(BaseModel):
    # Salary: leave floor unset to ignore salary entirely. currency is a symbol
    # or code used only for the prompt wording, e.g. "£", "$", "USD".
    salary_floor: int | None = None
    salary_currency: str = ""
    # Office: max days/week the candidate will work in-office. Unset = ignore
    # location/remote policy.
    max_office_days: int | None = None
    # Cap the score when the domain needs specialist expertise the candidate
    # lacks. 100 = no ceiling (the default; scoring is otherwise unconstrained).
    specialist_domain_ceiling: int = 100
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


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
    extra: dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    gates: list[GateConfig] = Field(default_factory=list)
    priority: PriorityConfig = Field(default_factory=PriorityConfig)
    sources: list[SourceConfig] = Field(default_factory=list)


def assessment_config_hash(cfg: Config) -> str:
    """Fingerprint of everything that determines a score's output.

    Changing any of these makes prior scores stale: rescore --all uses this to skip
    listings already scored under the current config instead of re-paying for them.
    """
    payload = {
        "scorer": cfg.ai.scorer,
        "bulk_model": cfg.ai.bulk_model,
        "bulk_reasoning": cfg.ai.bulk_reasoning,
        "scorer_repeats": cfg.ai.scorer_repeats,
        "gates": [g.model_dump() for g in cfg.gates],
        "priority": cfg.priority.model_dump(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


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

    return Config.model_validate(raw)
