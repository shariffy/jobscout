"""Listing/profile text helpers shared by extraction and prep.

The additive 0–100 BulkScorer lived here; gated assessment replaced it.
"""
from __future__ import annotations

import json

from .models import CandidateProfile, Listing


def _profile_text(profile: CandidateProfile) -> str:
    lines = [
        f"Seniority: {profile.seniority}",
        f"Summary: {profile.summary}",
        f"Skills: {', '.join(profile.skills)}",
        f"Domains: {', '.join(profile.domains)}",
        f"Must-haves: {', '.join(profile.must_haves)}",
        f"Nice-to-haves: {', '.join(profile.nice_to_haves)}",
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


def content_hash(listing: Listing) -> str:
    """Fingerprint of the exact text an extraction is computed from.

    Keys the extraction cache on the listing's *content*, not just its id: when a
    fuller JD is fetched and the description changes, this hash changes, so the stale
    extraction misses and the listing is re-read instead of silently reused.
    """
    import hashlib

    return hashlib.sha256(_listing_text(listing).encode()).hexdigest()[:16]


def _parse_score(text: str) -> dict:
    """Parse the first JSON object from model output (code fence / trailing prose ok)."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start = raw.find("{")
    if start != -1:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        return obj
    return json.loads(raw)
