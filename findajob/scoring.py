from __future__ import annotations

import json
from datetime import UTC, datetime

import anthropic

from .config import Config
from .models import CandidateProfile, Listing, Score

_SYSTEM_TEMPLATE = """\
You are a job-fit analyst. Score how well a job listing matches the candidate profile below.

Return ONLY a JSON object — no markdown, no prose — exactly:
{{
  "fit_score": <integer 0-100>,
  "breakdown": {{
    "title": <+5 to +20 — +18-20: exact preferred title; +12-17: close variant (e.g. Head vs Director); +5-11: weaker match (EM when Director preferred)>,
    "scope": <+5 to +20 — +18-20: full strategic autonomy, hands-on expected, first/senior eng leader; +12-17: mostly strategic, some autonomy gaps; +5-11: limited scope or IC-only>,
    "company": <+5 to +15 — +13-15: ideal size/stage/type (product, Series A-C, <200 people); +8-12: decent fit with minor gaps; +5-7: wrong size or stage>,
    "stack": <0 to +10 — +9-10: exact stack match (same technologies named explicitly); +6-8: strong overlap (3+ of candidate's core techs); +3-5: partial overlap (1-2 techs); +1-2: different stack but transferable; 0: no overlap or specialist tech required>,
    "domain": <-10 to 0 — 0: preferred domain or neutral; -3 to -5: not preferred but learnable (apply max -5 for these); -6 to -10: requires specialist domain knowledge candidate lacks>,
    "salary": <-3 to 0 — 0: salary stated and above threshold; -3: not stated (norm in UK — apply exactly -3); -20: stated below threshold>,
    "office": <-3 to 0 — 0: remote or hybrid clearly under 3 days; -3: minimum 3 days (borderline); -20: 4+ days or fully in-office confirmed>,
    "dealbreakers": <-50 to 0 — 0: none triggered; -50: hard dealbreaker confirmed (agency, managing-managers at scale, sponsorship needed, banned industry)>
  }},
  "rationale": "<2-3 sentences explaining the score>",
  "flags": ["<short tag>", ...]
}}

The fit_score must equal 50 + sum of all breakdown values (capped at 0–100).

Scoring guide:
- 90-100: Near-perfect — title, scope, company type, salary, and location all align
- 70-89: Strong — most criteria met, minor gaps
- 50-69: Partial — some good signals but meaningful gaps
- 30-49: Weak — doesn't fit well but no hard dealbreaker
- 0-29: Poor fit or a dealbreaker triggered (score 0-15 if dealbreaker confirmed)

Important interpretation rules:

Location:
- The "Location" field is scraped from a job card and may be inaccurate (e.g. "In-Office" when the
  full description says "Hybrid"). Always prefer the office/remote policy stated in the description
  over the Location field when they conflict.
- Office days: "minimum N days in office" means N days hybrid — treat it as N days exactly, not more.
  Only flag dealbreaker-office if the role explicitly requires 4+ days, "full time", "5 days/week", or
  "fully in-office". "Minimum 3 days" is hybrid at the candidate's stated maximum — flag hybrid-borderline
  but deduct at most 3 points; it is NOT a dealbreaker.

Org scale:
- Distinguish total company headcount from engineering team size. "Going from 10 to 100 people" refers
  to the whole company, not the engineering org. A Director/Head at a 10–100-person company typically
  leads a small team of ICs directly — that is NOT managing managers at scale.
- Generic startup phrases like "build and lead a world-class engineering organization", "scale our
  engineering team", "grow the engineering function", or "build and grow a team" do NOT mean managing
  managers — at small companies they mean being the first or most senior engineering leader.
- Only flag dealbreaker-managing-managers if the JD explicitly names managers, EMs, or team leads as
  direct reports, or uses phrases like "lead through other leaders", "manage engineering managers",
  or "multiple teams through managers". The bar for this flag should be high.

Domain:
- Do not score against a fixed whitelist of preferred domains. Instead assess adaptability: could an
  experienced engineering leader with this candidate's background realistically succeed and add value
  in this domain without deep specialist knowledge? Most product domains are accessible to a generalist
  senior engineering leader. Only deduct significantly (10+ points) if the domain requires specialist
  expertise the candidate clearly lacks (e.g. hardware/embedded, medical device compliance, quant
  trading algorithms). For domains that are simply "not their preference" but not a barrier, deduct
  at most 5 points and note it as a minor gap.

Salary:
- Salary not stated is the norm in UK tech hiring — the majority of listings omit it. Flag it with
  salary-not-stated but deduct at most 3 points. Do not treat missing salary the same as a confirmed
  low salary. Only trigger dealbreaker-salary if the salary IS explicitly stated and falls below the
  candidate's stated minimum.

Flag vocabulary (use what fits, invent concise ones for anything unusual):
  dealbreaker-agency, dealbreaker-salary, dealbreaker-office, dealbreaker-sponsorship, dealbreaker-industry
  dealbreaker-managing-managers
  title-strong, title-weak, title-mismatch
  salary-above-threshold, salary-below-threshold, salary-not-stated
  company-well-known, company-size-ok, company-too-large
  hybrid-ok, hybrid-borderline, office-heavy, remote-only
  domain-match, stack-match, scope-strategic, scope-ic-only, hands-on-expected

## Candidate Profile

{profile_text}
"""


def _profile_text(profile: CandidateProfile) -> str:
    lines = [
        f"Seniority: {profile.seniority}",
        f"Summary: {profile.summary}",
        f"Skills: {', '.join(profile.skills)}",
        f"Domains: {', '.join(profile.domains)}",
        f"Must-haves: {'; '.join(profile.must_haves)}",
        f"Nice-to-haves: {'; '.join(profile.nice_to_haves)}",
        f"Dealbreakers: {'; '.join(profile.dealbreakers)}",
    ]
    if profile.raw_goals:
        lines.append(f"\nFull goals:\n{profile.raw_goals}")
    return "\n".join(lines)


def _listing_text(listing: Listing) -> str:
    parts = [
        f"Title: {listing.title}",
        f"Company: {listing.company}",
    ]
    if listing.location:
        parts.append(f"Location: {listing.location}")
    if listing.description:
        parts.append(f"\nDescription:\n{listing.description}")
    parts.append(f"\nURL: {listing.url}")
    return "\n".join(parts)


def _parse_score(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


class BulkScorer:
    """Score listings against a candidate profile using Haiku with prompt caching."""

    def __init__(self, cfg: Config, profile: CandidateProfile) -> None:
        self._cfg = cfg
        self._profile = profile
        self._client = anthropic.Anthropic(api_key=cfg.ai.anthropic_api_key)
        self._system = _SYSTEM_TEMPLATE.format(profile_text=_profile_text(profile))

    def score(self, listing: Listing) -> Score:
        response = self._client.messages.create(
            model=self._cfg.ai.bulk_model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": self._system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Score this listing:\n\n{_listing_text(listing)}",
                }
            ],
        )

        data = _parse_score(response.content[0].text)

        breakdown = data.get("breakdown", {})
        if breakdown:
            # Compute score from breakdown to prevent holistic drift
            fit_score = max(0, min(100, 50 + sum(breakdown.values())))
        else:
            fit_score = int(data["fit_score"])

        return Score(
            listing_id=listing.id,
            fit_score=fit_score,
            rationale=data.get("rationale", ""),
            flags=data.get("flags", []),
            breakdown=breakdown,
            model=self._cfg.ai.bulk_model,
            tier="bulk",
            scored_at=datetime.now(UTC),
        )

    def score_batch(
        self,
        listings: list[Listing],
        progress_cb=None,
    ) -> list[Score]:
        scores = []
        for i, listing in enumerate(listings):
            score = self.score(listing)
            scores.append(score)
            if progress_cb:
                progress_cb(i + 1, len(listings), listing, score)
        return scores
