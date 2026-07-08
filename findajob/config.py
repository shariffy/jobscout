from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ProfileConfig(BaseModel):
    cv_path: str = "cv.pdf"
    goals: str = ""


class AIConfig(BaseModel):
    anthropic_api_key: str = Field(default="")
    bulk_model: str = "claude-haiku-4-5"
    deep_model: str = "claude-sonnet-4-6"
    prep_model: str = "claude-sonnet-4-6"
    fit_threshold: int = 70

    @model_validator(mode="after")
    def resolve_api_key(self) -> AIConfig:
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
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
    db_path: str = "findajob.db"


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
    extra: dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    sources: list[SourceConfig] = Field(default_factory=list)


_CONFIG_PATHS = ["config.toml", "~/.config/findajob/config.toml"]


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
