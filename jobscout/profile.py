from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

from .config import Config
from .llm import resolve_route, with_retries
from .models import CandidateProfile
from .scoring import _parse_score
from .store import _chmod_owner_only

_CACHE_FILENAME = "candidate_profile.json"

_SYSTEM = """\
You are an expert career analyst. Given a CV/resume and the candidate's stated goals, \
extract a structured profile that will be used to score job listings.

Return ONLY a JSON object — no markdown, no prose — with exactly these fields:
{
  "summary": "2-3 sentence overview of background and key strengths",
  "skills": ["specific skills, e.g. Node.js, team scaling, AWS — not generic terms"],
  "seniority": "e.g. Director / VP Engineering, Staff Engineer, CTO",
  "domains": ["industry or functional domains they have experience in"],
  "must_haves": ["things a role MUST have for them to apply"],
  "nice_to_haves": ["things that would make a role more attractive"],
  "dealbreakers": ["things that immediately disqualify a role"],
  "raw_goals": "candidate's stated goals, copied verbatim"
}

Be specific and concrete. "Node.js" beats "backend". "£150k+ salary" beats "good compensation".\
"""


def extract_cv_text(cv_path: str | Path) -> str:
    path = Path(cv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CV not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    elif suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported CV format: {suffix!r} — use .pdf or .md")


def build_profile(cfg: Config, force: bool = False) -> CandidateProfile:
    cache_path = Path(_CACHE_FILENAME)

    if cache_path.exists() and not force:
        data = json.loads(cache_path.read_text())
        return CandidateProfile(**data)

    cv_text = extract_cv_text(cfg.profile.cv_path)
    goals = cfg.profile.goals.strip()

    user_prompt = f"## CV\n\n{cv_text}\n\n## Candidate's stated goals\n\n{goals}"

    route = resolve_route(cfg, cfg.ai.deep_model)
    response = with_retries(
        lambda: route.client.chat.completions.create(
            model=route.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        ),
        label=cfg.ai.deep_model,
    )

    # Reasoning models keep their thinking in a separate field; the answer is in
    # message.content.
    raw = ((response.choices[0].message.content or "") if response.choices else "").strip()
    data = _parse_score(raw)
    # Inject raw goals if the model didn't copy them
    if not data.get("raw_goals") and goals:
        data["raw_goals"] = goals

    profile = CandidateProfile(**data)
    cache_path.write_text(json.dumps(data, indent=2))
    _chmod_owner_only(cache_path)
    return profile
