"""Helpers left in scoring.py after the additive BulkScorer was removed."""
from jobscout.scoring import _parse_score


def test_parse_score_plain_json():
    raw = '{"decision": "apply", "rationale": "Good match.", "flags": ["title-strong"]}'
    data = _parse_score(raw)
    assert data["decision"] == "apply"
    assert data["flags"] == ["title-strong"]


def test_parse_score_strips_markdown_fence():
    raw = '```json\n{"ok": true, "n": 72}\n```'
    data = _parse_score(raw)
    assert data["n"] == 72


def test_parse_score_tolerates_trailing_prose():
    raw = '```json\n{"n": 32, "breakdown": {"title": 8}}\n```\nHere is why.'
    data = _parse_score(raw)
    assert data["n"] == 32


def test_parse_score_tolerates_leading_prose_unfenced():
    raw = 'Sure, here is the score:\n{"n": 50, "breakdown": {}}'
    data = _parse_score(raw)
    assert data["n"] == 50
