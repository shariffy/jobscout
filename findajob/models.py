from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Listing(BaseModel):
    id: int | None = None
    hash: str = ""
    source_name: str
    title: str
    company: str
    location: str = ""
    url: str
    description: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def compute_hash(self) -> Listing:
        if not self.hash:
            payload = json.dumps(
                {"url": self.url, "title": self.title, "company": self.company},
                sort_keys=True,
            )
            self.hash = hashlib.sha256(payload.encode()).hexdigest()
        return self


class Score(BaseModel):
    id: int | None = None
    listing_id: int
    fit_score: int
    rationale: str
    flags: list[str] = Field(default_factory=list)
    breakdown: dict[str, int] = Field(default_factory=dict)
    model: str
    tier: str  # "bulk" | "deep"
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Application(BaseModel):
    id: int | None = None
    listing_id: int
    status: str = "shortlisted"  # shortlisted | applied | interviewing | offer | rejected | withdrawn
    applied_at: datetime | None = None
    chase_at: datetime | None = None
    notes: str = ""
    contacts: str = ""
    notion_page_id: str = ""
    prep_content: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateProfile(BaseModel):
    summary: str
    skills: list[str] = Field(default_factory=list)
    seniority: str = ""
    domains: list[str] = Field(default_factory=list)
    must_haves: list[str] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    dealbreakers: list[str] = Field(default_factory=list)
    raw_goals: str = ""
