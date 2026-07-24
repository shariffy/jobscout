from __future__ import annotations

import sys

from .config import Config
from .llm import extra_body_for, resolve_route, with_retries
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
    route = resolve_route(cfg, cfg.ai.prep_model)

    user_content = f"Candidate profile:\n{_profile_text(profile)}\n\n---\n\nJob listing:\n{_listing_text(listing)}"
    if score:
        tier = f" {score.tier_label}" if score.tier_label not in ("", "none") else ""
        user_content += f"\n\nAssessment: {score.decision.upper()}{tier} — {score.rationale}"

    # Server-side web search (when the provider offers one): the model decides
    # when to search, the provider runs it, and the grounded answer comes back in
    # message.content. Passed via extra_body so it isn't touched by the OpenAI
    # SDK's function-tool validation — each provider's shape differs (OpenRouter/
    # Requesty hang a tool off `tools[]`; Google nests its own grounding config
    # under a `google` key), so the provider entry owns its whole extra_body slice.
    # Anthropic direct has no such tool here — the brief runs ungrounded on that route.
    extra_body = extra_body_for(route.provider, None)
    if route.provider.web_search_extra_body is not None:
        extra_body.update(route.provider.web_search_extra_body)
    else:
        print(
            f"[prep] {route.provider_name} has no server-side web search — "
            "generating an ungrounded brief (the 'who to contact' section may be weaker).",
            file=sys.stderr,
        )

    response = with_retries(
        lambda: route.client.chat.completions.create(
            model=route.model,
            max_tokens=6000,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            extra_body=extra_body,
        ),
        label=cfg.ai.prep_model,
    )

    return (response.choices[0].message.content or "") if response.choices else ""
