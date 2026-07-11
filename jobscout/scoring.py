from __future__ import annotations

import json
from datetime import UTC, datetime

import anthropic

from .config import Config
from .models import CandidateProfile, Listing, Score

_SYSTEM_TEMPLATE = """\
You are a job-fit analyst. Score how well a job listing matches the candidate profile below.
All judgements about what the candidate wants, is qualified for, and will not accept must come
from the CANDIDATE PROFILE — do not assume a particular industry, seniority, or field.

Return ONLY a JSON object — no markdown, no prose — exactly:
{{
  "fit_score": <integer 0-100>,
  "breakdown": {{
    "title": <0 to +{w_title} — score by how the role's title ranks among the candidate's preferred titles. If the candidate gives a ranked list, a top-ranked title scores near +{w_title}, a mid-ranked one about half, and a title at the bottom of the list or absent from it scores low (a quarter of +{w_title} or less). Do NOT give a bottom-ranked or non-preferred title more than half of +{w_title}>,
    "scope": <0 to +{w_scope} — judge against the CAREER TRACK the candidate wants (see the Track rule below), not just autonomy. Highest when the role is on the candidate's desired track at the level they want. Deduct toward 0 when the role is over-scoped (materially more scale/leadership than they have held) OR off-track (e.g. an individual-contributor role when the candidate wants a leadership role, or vice versa) — a role can offer strong technical autonomy and still be a scope mismatch>,
    "company": <0 to +{w_company} — highest when company size, stage, and type match the candidate's stated preferences; lower for clear mismatches>,
    "stack": <0 to +{w_stack} — how well the skills, tools, or qualifications the role requires match the candidate's. Highest for strong/explicit overlap; low for a different-but-transferable background; near 0 for no overlap. If the role hard-requires a specific skill the candidate clearly lacks and it is central to the job (not just "a plus"), keep this low and flag skills-mismatch-required>,
    "domain": <-{w_domain} to 0 — 0 for the candidate's preferred domains OR any neutral domain where someone with their background succeeds without specialist knowledge (default to 0 unless there is a specific reason not to); a moderate negative for a domain the candidate has reason to avoid or that needs real specialist adaptation; the largest negative only when the role explicitly gates on domain experience the candidate lacks>,
    "salary": {salary_scale},
    "office": {office_scale},
    "dealbreakers": <-{w_dealbreakers} to 0 — 0 if none of the candidate's stated dealbreakers (see profile) are triggered; -{w_dealbreakers} if any hard dealbreaker from the profile is confirmed in the listing>
  }},
  "rationale": "<2-3 sentences explaining the score>",
  "flags": ["<short tag>", ...]
}}

The fit_score must equal 50 + sum of all breakdown values (capped at 0–100).

Scoring guide:
- 95-100: Exceptional — reserve for roles where EVERY positive is near-maximum and no negative
  applies. Do not award 95+ if any criterion is absent or uncertain.
- 85-94: Near-perfect — title and scope are strong, at most one minor gap.
- 70-84: Strong — most criteria met, two or more minor gaps present.
- 50-69: Partial — some good signals but meaningful gaps.
- 30-49: Weak — doesn't fit well but no hard dealbreaker.
- 0-29: Poor fit or a dealbreaker triggered (score 0-15 if a dealbreaker is confirmed).
{ceiling_rule}
Important interpretation rules:

Location and office presence:
- The "Location" field is scraped from a job card and may be inaccurate (e.g. "In-Office" when the
  full description says "Hybrid"). Always prefer the office/remote policy stated in the description
  over the Location field when they conflict.
- Read the description for in-office signals even when there is no explicit policy line: phrasing
  like "based in <city>", "we're in the office", a named HQ presented as central to how the team
  works, or an emphasis on co-location with no mention of remote/hybrid all indicate a meaningful
  in-office expectation. When such signals imply more in-office presence than the candidate's stated
  maximum, apply the office penalty (and the office dealbreaker if the profile makes it one).
- But do not over-trigger: when the listing is genuinely ambiguous about how many office days are
  required, treat it as borderline (the middle office deduction), not an automatic dealbreaker.
- "Minimum N days in office" where N equals the candidate's stated maximum office days counts as AT
  the maximum (the borderline middle deduction), not above it. Do not escalate to the full office
  penalty unless the listing states more than N days or names additional mandatory in-office
  expectations — a strong role must not be tanked by an at-maximum office requirement.

Career track (leadership vs individual contributor):
- Infer from the candidate's preferred/ranked titles whether they primarily want a leadership/management
  track (Head, Director, VP, Engineering Manager) or an individual-contributor track (Staff, Principal,
  Senior Engineer). Score BOTH title and scope relative to that track.
- If the candidate wants a leadership track, an individual-contributor role (Staff/Principal/Senior
  engineer, or a "Tech Lead" with no team, direct reports, or roadmap ownership) is a scope mismatch:
  score scope low — roughly a third of the maximum or less — and flag scope-track-mismatch, even if the
  role offers strong technical autonomy. Do not award high scope to an IC role for a leadership-seeking
  candidate just because it is senior or hands-on.
- Exception: honour any exception the candidate states in their profile. For example, if the candidate
  says they will accept an IC role at a well-known / top-tier consumer brand, then for such a company
  treat the IC role as on-track and score scope normally.
- Apply the reverse for an IC-track candidate faced with a pure people-management role.

Scope and scale:
- Distinguish total company headcount from team size. Generic growth phrases ("scale the team",
  "build a world-class organisation", "grow the function") do NOT by themselves mean the role
  exceeds the candidate's experience — at smaller companies they usually mean being the first or
  most senior person in that function. Only treat a role as over-scoped when the listing clearly
  requires a scale of leadership or responsibility the candidate's profile says they have not held.
- Concretely, treat a role as over-scoped (drive scope toward 0 and flag scope-over-scoped) when it
  requires leading other engineering leaders/managers, owning a multi-team or org-wide function, or
  scaling an organisation materially larger than the candidate has run — e.g. a VP/CTO-level remit
  for a candidate targeting Head/Director. A title one step above the candidate's target that also
  carries organisation-building scope is a scope mismatch, not a stretch role.

Domain:
- Do not score against a fixed whitelist. Assess adaptability: could this candidate realistically
  succeed and add value in this domain without deep specialist knowledge? Most domains are accessible.
  Deduct heavily only when the domain genuinely requires specialist expertise the candidate lacks.
  For domains that are merely "not their preference" but not a barrier, deduct only a little.

Dealbreakers:
- The candidate's dealbreakers are listed in the profile. Trigger the dealbreakers penalty only when
  the listing clearly confirms one of them; do not infer a dealbreaker from ambiguous wording.
- Function/track mismatch is dealbreaker-class: if the role is not in the candidate's field at all
  (e.g. a sales, marketing, product-management, or other non-engineering role for an engineering
  candidate), OR is a hard career-track mismatch that falls outside every exception the candidate
  states, treat it like a confirmed dealbreaker — apply the full dealbreakers penalty and score
  0–15. A fundamentally wrong role must not float near 50 just because no positive dimension scored
  strongly negative.

Flag vocabulary (use what fits, invent concise ones for anything unusual):
  dealbreaker-<name> for any confirmed dealbreaker from the profile
  title-strong, title-weak, title-mismatch
  salary-above-threshold, salary-below-threshold, salary-not-stated
  company-size-ok, company-too-large, company-stage-mismatch
  remote-only, hybrid-ok, hybrid-borderline, office-heavy
  domain-match, domain-gated, skills-match, skills-mismatch-required
  scope-match, scope-over-scoped, scope-under-scoped, scope-track-mismatch

## Candidate Profile

{profile_text}
"""


def _build_system(cfg: Config, profile: CandidateProfile) -> str:
    """Render the scoring system prompt from config weights and thresholds."""
    w = cfg.scoring.weights
    s = cfg.scoring

    if s.salary_floor is not None:
        floor = f"{s.salary_currency}{s.salary_floor:,}"
        salary_scale = (
            f"<-20 to 0 — 0: salary stated at or above {floor}; "
            f"-{w.salary}: salary not stated (common; deduct only this much, do not treat as low); "
            f"-20: stated below {floor}>"
        )
    else:
        salary_scale = "<always 0 — no salary threshold set; ignore compensation>"

    if s.max_office_days is not None:
        n = s.max_office_days
        office_scale = (
            f"<-20 to 0 — 0: remote, or hybrid requiring fewer than {n} day(s)/week in office; "
            f"-{w.office}: hybrid at exactly {n} day(s) (the candidate's stated maximum — borderline, "
            f"not a dealbreaker); -20: requires more than {n} day(s)/week or is fully in-office>"
        )
    else:
        office_scale = "<always 0 — no office-day preference set; ignore location/remote policy>"

    if s.specialist_domain_ceiling < 100:
        ceiling_rule = (
            f"\nHard ceiling: if the domain requires meaningful specialist adaptation the candidate\n"
            f"lacks, the score cannot exceed {s.specialist_domain_ceiling} regardless of other strengths.\n"
        )
    else:
        ceiling_rule = "\n"

    return _SYSTEM_TEMPLATE.format(
        w_title=w.title,
        w_scope=w.scope,
        w_company=w.company,
        w_stack=w.stack,
        w_domain=w.domain,
        w_dealbreakers=w.dealbreakers,
        salary_scale=salary_scale,
        office_scale=office_scale,
        ceiling_rule=ceiling_rule,
        profile_text=_profile_text(profile),
    )


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
    # Strip a leading markdown code fence if the model wrapped its JSON in one.
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    # Decode the first JSON object, tolerating any prose or extra data that some
    # models (e.g. Haiku) append after it — json.loads() alone rejects trailing text.
    start = raw.find("{")
    if start != -1:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        return obj
    return json.loads(raw)


class BulkScorer:
    """Score listings against a candidate profile using Haiku with prompt caching."""

    def __init__(self, cfg: Config, profile: CandidateProfile) -> None:
        self._cfg = cfg
        self._profile = profile
        self._client = anthropic.Anthropic(api_key=cfg.ai.anthropic_api_key)
        self._system = _build_system(cfg, profile)

    def score(self, listing: Listing) -> Score:
        messages = [
            {
                "role": "user",
                "content": f"Score this listing:\n\n{_listing_text(listing)}",
            }
        ]
        system = [
            {
                "type": "text",
                "text": self._system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        last_exc: Exception | None = None
        for attempt in range(2):
            response = self._client.messages.create(
                model=self._cfg.ai.bulk_model,
                max_tokens=4096,
                system=system,
                messages=messages,
            )
            # Reasoning models (e.g. Sonnet 5, Opus 4.8) emit a thinking block
            # before the answer, so content[0] isn't necessarily the text — take
            # the first text block. Also skips any non-text (tool) blocks.
            raw = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
            try:
                data = _parse_score(raw)
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    # Retry once; ask the model to emit only the JSON
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": "Reply with only the JSON object, no other text."},
                    ]
        else:
            raise ValueError(f"Could not parse score after 2 attempts. Last raw: {raw!r}") from last_exc

        breakdown = data.get("breakdown", {})
        if breakdown:
            fit_score = max(0, min(100, 50 + sum(breakdown.values())))
            # Enforce the specialist-domain ceiling: when the model penalised the
            # domain meaningfully (-5 or worse), the role needs specialist
            # adaptation the candidate lacks, so cap the score. Neutral domains
            # score 0 and are unaffected. Disabled when the ceiling is 100.
            ceiling = self._cfg.scoring.specialist_domain_ceiling
            if ceiling < 100 and breakdown.get("domain", 0) <= -5:
                fit_score = min(fit_score, ceiling)
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
