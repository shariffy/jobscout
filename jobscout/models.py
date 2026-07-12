from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


def content_key(company: str, title: str) -> str:
    """Normalised identity for a listing: same (company, title) => same key.

    Keyed on company + title rather than URL so the same job dedups across
    sources (e.g. LinkedIn vs Built In) and across searches whose URLs carry
    different tracking parameters. Lowercased, punctuation/whitespace collapsed.
    Parenthetical suffixes are stripped from the company (Built In writes
    "Light (light.inc)" where LinkedIn writes "Light") but kept in the title,
    where "(Platform)" or "(Frontend)" can distinguish genuinely different roles.
    """
    def norm(s: str, strip_parens: bool = False) -> str:
        s = (s or "").lower()
        if strip_parens:
            s = re.sub(r"\([^)]*\)", " ", s)  # drop "(light.inc)", "(zego.com)"
        return re.sub(r"[^a-z0-9]+", " ", s).strip()

    return f"{norm(company, strip_parens=True)}::{norm(title)}"


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
            self.hash = hashlib.sha256(content_key(self.company, self.title).encode()).hexdigest()
        return self


class Score(BaseModel):
    id: int | None = None
    listing_id: int
    # DEPRECATED migration glue: derived from decision/tier by the gated scorer,
    # judged directly only by the legacy additive scorer. apply <=> fit_score >= 70.
    fit_score: int
    rationale: str
    flags: list[str] = Field(default_factory=list)
    breakdown: dict[str, int] = Field(default_factory=dict)
    model: str
    tier: str  # "bulk" | "deep"
    # First-class assessment fields (gated scorer). Empty/None on legacy rows
    # until the store backfills decision from fit_score.
    decision: str = ""  # "apply" | "no" | "" (unassessed legacy row)
    priority: int | None = None  # sort key within the apply set; lower = do first
    tier_label: str = ""  # "T1".."T4" | "none" | "" (distinct from bulk/deep tier)
    gate_results: list[dict[str, Any]] = Field(default_factory=list)
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
