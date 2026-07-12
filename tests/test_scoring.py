"""Scoring tests — OpenRouter (OpenAI-compatible) client mocked, no real API calls."""
import json
from unittest.mock import MagicMock, patch

from jobscout.config import AIConfig, Config, StoreConfig, ProfileConfig
from jobscout.models import CandidateProfile, Listing
from jobscout.scoring import BulkScorer, _parse_score


def make_cfg(**ai_kwargs) -> Config:
    cfg = Config()
    defaults = dict(openrouter_api_key="test-key", bulk_model="anthropic/claude-haiku-4-5")
    defaults.update(ai_kwargs)
    cfg.ai = AIConfig(**defaults)
    return cfg


def make_profile() -> CandidateProfile:
    return CandidateProfile(
        summary="Senior engineering leader.",
        skills=["Node.js", "AWS", "team leadership"],
        seniority="Director of Engineering",
        domains=["music tech", "ecommerce"],
        must_haves=["min £110k salary", "hybrid London"],
        nice_to_haves=["music domain", "Node.js stack"],
        dealbreakers=["agency work", "salary below £110k", "office >3 days/week"],
        raw_goals="Looking for Head of Engineering at a product company.",
    )


def make_listing(**kwargs) -> Listing:
    defaults = dict(
        id=1,
        source_name="test",
        title="Head of Engineering",
        company="Acme Music",
        location="London / Hybrid",
        url="https://example.com/jobs/1",
        description="Lead our 15-person engineering team. £130k–£160k. 2 days/week in office.",
    )
    defaults.update(kwargs)
    return Listing(**defaults)


def mock_response(fit_score: int, rationale: str, flags: list[str]) -> MagicMock:
    """Mimic an OpenRouter chat.completions response: message.content holds the JSON."""
    message = MagicMock()
    message.content = json.dumps({"fit_score": fit_score, "rationale": rationale, "flags": flags})
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# --- _parse_score ---

def test_parse_score_plain_json():
    raw = '{"fit_score": 85, "rationale": "Good match.", "flags": ["title-strong"]}'
    data = _parse_score(raw)
    assert data["fit_score"] == 85
    assert data["flags"] == ["title-strong"]


def test_parse_score_strips_markdown_fence():
    raw = '```json\n{"fit_score": 72, "rationale": "Decent.", "flags": []}\n```'
    data = _parse_score(raw)
    assert data["fit_score"] == 72


def test_parse_score_tolerates_trailing_prose():
    # Haiku sometimes appends explanation after the JSON, which json.loads rejects.
    raw = '```json\n{"fit_score": 32, "breakdown": {"title": 8}}\n```\nHere is why I scored it 32.'
    data = _parse_score(raw)
    assert data["fit_score"] == 32


def test_parse_score_tolerates_leading_prose_unfenced():
    raw = 'Sure, here is the score:\n{"fit_score": 50, "breakdown": {}}'
    data = _parse_score(raw)
    assert data["fit_score"] == 50


# --- BulkScorer ---

@patch("jobscout.llm.openai.OpenAI")
def test_bulk_scorer_returns_score(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_response(
        88, "Strong title and salary match.", ["title-strong", "salary-above-threshold", "hybrid-ok"]
    )

    scorer = BulkScorer(make_cfg(), make_profile())
    score = scorer.score(make_listing())

    assert score.fit_score == 88
    assert score.tier == "bulk"
    assert score.model == "anthropic/claude-haiku-4-5"
    assert "title-strong" in score.flags
    assert score.listing_id == 1


@patch("jobscout.llm.openai.OpenAI")
def test_bulk_scorer_uses_openrouter_base_url(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_response(75, "Good.", [])

    BulkScorer(make_cfg(), make_profile()).score(make_listing())

    init_kwargs = mock_openai_cls.call_args.kwargs
    assert init_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert init_kwargs["api_key"] == "test-key"


@patch("jobscout.llm.openai.OpenAI")
def test_bulk_scorer_sends_system_and_user_messages(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_response(75, "Good.", [])

    scorer = BulkScorer(make_cfg(), make_profile())
    scorer.score(make_listing())

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # cost accounting is always requested; reasoning omitted when unset
    assert call_kwargs["extra_body"]["usage"] == {"include": True}
    assert "reasoning" not in call_kwargs["extra_body"]


@patch("jobscout.llm.openai.OpenAI")
def test_bulk_scorer_passes_reasoning_config(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_response(75, "Good.", [])

    cfg = make_cfg(bulk_reasoning={"effort": "low"})
    BulkScorer(cfg, make_profile()).score(make_listing())

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["reasoning"] == {"effort": "low"}


@patch("jobscout.llm.openai.OpenAI")
def test_bulk_scorer_dealbreaker_low_score(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_response(
        8, "Agency work — dealbreaker.", ["dealbreaker-agency"]
    )

    scorer = BulkScorer(make_cfg(), make_profile())
    score = scorer.score(make_listing(title="Senior Consultant", company="Big Agency Ltd"))

    assert score.fit_score == 8
    assert "dealbreaker-agency" in score.flags


@patch("jobscout.llm.openai.OpenAI")
def test_score_batch(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = mock_response(80, "Good.", ["title-strong"])

    listings = [
        make_listing(id=1, url="https://example.com/1", title="Head of Eng"),
        make_listing(id=2, url="https://example.com/2", title="VP Engineering"),
    ]

    scorer = BulkScorer(make_cfg(), make_profile())
    scores = scorer.score_batch(listings)

    assert len(scores) == 2
    assert mock_client.chat.completions.create.call_count == 2
