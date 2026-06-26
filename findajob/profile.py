from __future__ import annotations

import json
from pathlib import Path

import anthropic
from pypdf import PdfReader

from .config import Config
from .models import CandidateProfile

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

    client = anthropic.Anthropic(api_key=cfg.ai.anthropic_api_key)
    response = client.messages.create(
        model=cfg.ai.deep_model,
        max_tokens=2048,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    # Inject raw goals if Claude didn't copy them
    if not data.get("raw_goals") and goals:
        data["raw_goals"] = goals

    profile = CandidateProfile(**data)
    cache_path.write_text(json.dumps(data, indent=2))
    return profile
