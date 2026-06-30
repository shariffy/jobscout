from __future__ import annotations

import anthropic

from .config import Config
from .models import CandidateProfile, Listing, Score
from .scoring import _listing_text, _profile_text

_SYSTEM = """\
You are an expert job-search coach helping a candidate prepare a strong application for a specific role.

Given the candidate profile and job listing below, produce a structured prep brief with four sections:

## Positioning
2–3 sentences on how the candidate should frame their background for this specific role. \
What's the strongest angle? What's the narrative thread that connects their experience to what the company needs?

## Gaps to address
Bullet list of the 1–3 most significant gaps or concerns the hiring team might raise, \
and one sentence each on how the candidate can pre-empt or mitigate them.

## Who to reach out to
The role(s) / seniority level(s) worth targeting at this company for a warm intro or connection \
(e.g. "hiring manager", "CTO", "engineering team"). \
Include a draft outreach message (LinkedIn DM or short email) the candidate can adapt. \
Keep it under 100 words — specific, not generic.

## Interview prep
5 likely interview topics or questions for this role, with one bullet each on what to emphasise in the answer.

Be direct and specific. Use the actual company name and role title. Do not pad with generic advice.
"""


def generate_prep(
    cfg: Config,
    listing: Listing,
    profile: CandidateProfile,
    score: Score | None = None,
) -> str:
    client = anthropic.Anthropic(api_key=cfg.ai.anthropic_api_key)

    user_content = f"Candidate profile:\n{_profile_text(profile)}\n\n---\n\nJob listing:\n{_listing_text(listing)}"
    if score:
        user_content += f"\n\nFit score: {score.fit_score}/100 — {score.rationale}"

    response = client.messages.create(
        model=cfg.ai.prep_model,
        max_tokens=2048,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text
