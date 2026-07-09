from __future__ import annotations

import anthropic

from .config import Config
from .models import CandidateProfile, Listing, Score
from .scoring import _listing_text, _profile_text

_SYSTEM = """\
You are an expert job-search coach. You have access to a web search tool — use it to look up \
real people at the company (LinkedIn, company website, Crunchbase, press releases) before writing \
the "Who to contact" section. Search for the company name plus role titles like CTO, VP Engineering, \
Head of Engineering, or the hiring manager for this specific role.

Given the candidate profile and job listing, produce a prep brief with exactly these six sections. \
Be specific to this role and company — no generic advice.

## CV adjustments
Bullet list of specific changes to make to the CV before applying:
- Reorder or reword existing bullets to mirror the JD's language
- Surface experience that's relevant but currently buried
- Flag anything to remove or de-emphasise for this application
Keep each bullet actionable: "Change X to Y" or "Move Z to the top of your most recent role".

## Cover letter
3–4 bullet points on what to lead with and highlight. Each point should name a specific \
experience or achievement from the candidate's background and explain why it maps to what \
this company needs. Include the strongest opening hook as a draft sentence.

## Hiring process
Based on the company (stage, size, known culture), describe what interview process to expect: \
number of stages, likely format (take-home, system design, leadership interview, values interview, \
CEO chat, etc.), and rough timeline. Note any company-specific quirks if known.

## Who to contact
Use your web search results to name actual people at this company worth reaching out to — \
the hiring manager if identifiable, the CTO, VP/Head of Engineering, or a relevant engineering leader. \
For each person: name, title, and a 2–3 sentence LinkedIn DM the candidate can send today. \
Be specific — reference the role, a genuine connection point, and a clear ask. Under 80 words per message.

## Study plan
Concrete things to do before the first interview, ordered by priority. Include:
- Technical topics to brush up on (specific to this stack or domain)
- Company/product research (what to read, what to understand about their business)
- Stories to prepare (which past projects map to their likely competency questions)
- Any practical prep (take-home format guess, system design topics for this domain)
Frame as a realistic plan — what to do today, what can wait.

## Interview questions to expect
6–8 likely questions, split between technical/system-design and leadership/behavioural. \
For each, one bullet on what to emphasise in the answer given this candidate's background.
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
        max_tokens=3000,
        system=_SYSTEM,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        messages=[{"role": "user", "content": user_content}],
    )

    # Collect all text blocks (web search results are tool_result blocks, skip them)
    return "\n\n".join(
        block.text for block in response.content if block.type == "text"
    )
