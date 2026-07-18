"""Gated assessment tests — pure-Python gate/priority engines plus a mocked
end-to-end GatedScorer run. No real API calls; this is where gate robustness
is locked in (each gate has a synthetic listing that must trip it)."""
import json
from unittest.mock import MagicMock, patch

import pytest

from jobscout.assess import (
    Assessment,
    Extraction,
    FeatureValue,
    GatedScorer,
    RuleVerdict,
    _apply_match_keywords,
    assess,
    assign_tier,
    build_extract_system,
    build_scorer,
    consensus_assessment,
    consensus_locked,
    evaluate_gate,
    parse_extraction,
)
from jobscout.config import AIConfig, Config
from jobscout.gates import GateConfig
from jobscout.llm import _client_cache
from jobscout.models import Listing
from jobscout.scoring import BulkScorer, content_hash

# --- fixtures ---------------------------------------------------------------

GATES = [
    dict(name="office", feature="office_days_per_week", op="<=", value=3, on_unknown="pass_flag"),
    dict(name="salary_floor", feature="salary_gbp", op=">=", value=110000),
    dict(name="industry", feature="industry", op="not_in", value=["crypto", "gambling"]),
    dict(name="no_blocked_stack",
         rule="Reject only if Java/JVM, Go, Rust, Kotlin, Scala, Ruby, PHP, C/C++, or C# is a hard requirement."),
    dict(name="no_agency", rule="Reject only if pure consulting/agency work."),
]

TIERS = [
    dict(name="T1", titles=["VP of Engineering", "Head of Engineering",
                            "Director of Engineering", "Senior Engineering Manager"],
         substances=["leadership"]),
    dict(name="T2", titles=["Founding Engineer", "Technical Lead"], substances=["founding"]),
    dict(name="T3", titles=["Engineering Manager"], substances=["management"]),
    dict(name="T4", titles=["Principal Engineer", "Staff Engineer"], substances=["ic"]),
]


def make_cfg(**overrides) -> Config:
    _client_cache.clear()  # each test gets a fresh mocked client per provider
    data = {
        "ai": {"api_keys": {"openrouter": "test-key"}, "bulk_model": "test/model", "scorer": "gated"},
        "gates": GATES,
        "priority": {"tiers": TIERS, "unknown_penalty": 2},
    }
    data.update(overrides)
    return Config.model_validate(data)


def make_listing(title="Head of Engineering", **kwargs) -> Listing:
    defaults = dict(
        id=1, source_name="test", title=title, company="Acme",
        url="https://example.com/jobs/1", description="A role.",
    )
    defaults.update(kwargs)
    return Listing(**defaults)


def fv(value, confidence=1.0, evidence="quoted") -> FeatureValue:
    return FeatureValue(value=value, confidence=confidence, evidence=evidence)


def extraction(features=None, rules=None, summary="A role at Acme.") -> Extraction:
    return Extraction(features=features or {}, rules=rules or {}, summary=summary)


# --- config validation -------------------------------------------------------

def test_gate_config_rejects_unknown_feature():
    with pytest.raises(ValueError, match="unknown feature"):
        GateConfig(name="bad", feature="nonexistent", op="==", value=1)


def test_gate_config_rejects_both_forms():
    with pytest.raises(ValueError, match="not both"):
        GateConfig(name="bad", feature="salary_gbp", op=">=", value=1, rule="Reject if x.")


def test_gate_config_rejects_neither_form():
    with pytest.raises(ValueError):
        GateConfig(name="bad")


def test_gate_config_rejects_bad_op():
    with pytest.raises(ValueError, match="op must be"):
        GateConfig(name="bad", feature="salary_gbp", op="~=", value=1)


def test_gate_config_rejects_match_keywords_on_rule_gate():
    with pytest.raises(ValueError, match="match_keywords"):
        GateConfig(
            name="no_blocked_stack",
            rule="Reject only if Java is a hard requirement.",
            match_keywords=["java"],
        )


# --- structured gate evaluation ----------------------------------------------

def gate(name) -> GateConfig:
    return next(g for g in make_cfg().gates if g.name == name)


def test_structured_gate_passes():
    r = evaluate_gate(gate("office"), {"office_days_per_week": fv(2)}, {})
    assert r.status == "pass"


def test_structured_gate_at_boundary_passes():
    # "minimum 3 days" with a 3-day maximum is AT the limit, not over it.
    r = evaluate_gate(gate("office"), {"office_days_per_week": fv(3)}, {})
    assert r.status == "pass"


def test_structured_gate_fails():
    r = evaluate_gate(gate("office"), {"office_days_per_week": fv(5)}, {})
    assert r.status == "fail"
    assert "office_days_per_week" in r.reason


def test_missing_feature_is_unknown():
    r = evaluate_gate(gate("office"), {}, {})
    assert r.status == "unknown"


def test_null_value_is_unknown():
    r = evaluate_gate(gate("salary_floor"), {"salary_gbp": fv(None)}, {})
    assert r.status == "unknown"


def test_low_confidence_value_is_unknown():
    # A weak detection must never hard-fail a gate (the false-dealbreaker fix).
    r = evaluate_gate(gate("office"), {"office_days_per_week": fv(5, confidence=0.4)}, {})
    assert r.status == "unknown"


def test_uncomparable_value_is_unknown():
    r = evaluate_gate(gate("office"), {"office_days_per_week": fv("hybrid")}, {})
    assert r.status == "unknown"


def test_not_in_gate_fails_on_member_case_insensitive():
    r = evaluate_gate(gate("industry"), {"industry": fv("Crypto")}, {})
    assert r.status == "fail"


def test_not_in_gate_passes_on_non_member():
    r = evaluate_gate(gate("industry"), {"industry": fv("music")}, {})
    assert r.status == "pass"


def test_no_blocked_stack_rule_fail():
    verdicts = {"no_blocked_stack": RuleVerdict(verdict="fail", confidence=0.95, evidence="must have Go")}
    r = evaluate_gate(gate("no_blocked_stack"), {}, verdicts)
    assert r.status == "fail"


def test_no_blocked_stack_rule_pass():
    verdicts = {"no_blocked_stack": RuleVerdict(verdict="pass", confidence=0.9)}
    r = evaluate_gate(gate("no_blocked_stack"), {}, verdicts)
    assert r.status == "pass"


def test_industry_private_equity_fails():
    industry_gate = GateConfig(
        name="industry", feature="industry", op="not_in",
        value=["crypto", "gambling", "private_equity"],
    )
    r = evaluate_gate(industry_gate, {"industry": fv("private_equity")}, {})
    assert r.status == "fail"


def test_match_keywords_private_equity_trips_industry_gate():
    industry_gate = GateConfig(
        name="industry", feature="industry", op="not_in",
        value=["crypto", "gambling", "private_equity"],
        match_keywords=["private equity"],
        keyword_value="private_equity",
    )
    listing = make_listing(
        description="Software for the world's leading investment firms across private equity, VC, credit.",
    )
    ext = extraction(features={"industry": fv(None)})
    _apply_match_keywords([industry_gate], listing, ext)
    r = evaluate_gate(industry_gate, ext.features, {})
    assert r.status == "fail"
    assert ext.features["industry"].value == "private_equity"


def test_assess_ohme_jvm_soft_mention_passes_when_rule_passes():
    cfg = make_cfg(gates=GATES + [
        dict(name="industry", feature="industry", op="not_in",
             value=["crypto", "gambling", "private_equity"]),
    ])
    listing = make_listing(
        title="Head of Engineering",
        company="Ohme",
        description="Demonstrable knowledge in cloud/distributed systems ideally in JVM at high-scale.",
    )
    e = clean_extraction()
    e.rules["no_blocked_stack"] = RuleVerdict(
        verdict="pass", confidence=0.9,
        evidence="ideally in JVM is a soft mention, not a hard requirement",
    )
    a = assess(cfg, listing, e)
    assert a.decision == "apply"
    assert not any(f == "gate-fail-no_blocked_stack" for f in a.flags)


def test_assess_malt_sem_hands_on_ideally_java_fails():
    """Malt #178: hands-on SEM with 'ideally java/Kotlin' must still reject."""
    cfg = make_cfg()
    listing = make_listing(
        title="Senior Engineering Manager - Compliance",
        company="Malt",
        description=(
            "Hands-on Manager: You maintain your technical mentorship through code reviews "
            "and architecture discussions. You likely have a strong backend foundation "
            "(ideally in java/Kotlin) and a passion for engineering excellence."
        ),
    )
    e = clean_extraction()
    e.rules["no_blocked_stack"] = RuleVerdict(
        verdict="fail", confidence=0.9,
        evidence="hands-on EM with strong backend foundation ideally in java/Kotlin",
    )
    a = assess(cfg, listing, e)
    assert a.decision == "no"
    assert "gate-fail-no_blocked_stack" in a.flags


def test_assess_gener8_vp_team_stack_description_passes():
    """Gener8 #124: Golang/Kotlin describe the team stack; VP hands-on-when-needed is not EM case."""
    cfg = make_cfg()
    listing = make_listing(
        title="VP of Engineering",
        company="Gener8",
        description=(
            "On the frontend we use React, Next.js and TypeScript. "
            "On the backend we use Golang to serve RESTful APIs. "
            "Our mobile apps have been built natively using Swift and Kotlin. "
            "Providing technical leadership and able to be hands on and in the detail when needed. "
            "Reviewing other engineer's code, making suggestions. "
            "7+ years experience as a software engineer working with native mobile or web technologies. "
            "3+ years experience in an engineering leadership role."
        ),
    )
    e = clean_extraction()
    e.rules["no_blocked_stack"] = RuleVerdict(
        verdict="pass", confidence=0.9,
        evidence="Golang/Kotlin are team stack; VP may review code but no backend-foundation requirement on candidate",
    )
    a = assess(cfg, listing, e)
    assert a.decision == "apply"
    assert not any(f == "gate-fail-no_blocked_stack" for f in a.flags)


def test_salary_gate_passes_at_floor():
    r = evaluate_gate(gate("salary_floor"), {"salary_gbp": fv(110000)}, {})
    assert r.status == "pass"


def test_salary_gate_fails_below_floor():
    r = evaluate_gate(gate("salary_floor"), {"salary_gbp": fv(90000)}, {})
    assert r.status == "fail"


# --- NL rule gate evaluation ---------------------------------------------------

def test_rule_gate_fail_verdict():
    verdicts = {"no_agency": RuleVerdict(verdict="fail", confidence=0.95)}
    r = evaluate_gate(gate("no_agency"), {}, verdicts)
    assert r.status == "fail"


def test_rule_gate_low_confidence_fail_is_unknown():
    verdicts = {"no_agency": RuleVerdict(verdict="fail", confidence=0.5)}
    r = evaluate_gate(gate("no_agency"), {}, verdicts)
    assert r.status == "unknown"


def test_rule_gate_pass_verdict():
    verdicts = {"no_agency": RuleVerdict(verdict="pass", confidence=0.9)}
    r = evaluate_gate(gate("no_agency"), {}, verdicts)
    assert r.status == "pass"


def test_rule_gate_missing_verdict_is_unknown():
    r = evaluate_gate(gate("no_agency"), {}, {})
    assert r.status == "unknown"


# --- decision + on_unknown policy ---------------------------------------------

def clean_extraction() -> Extraction:
    """Everything known and passing."""
    return extraction(
        features={
            "office_days_per_week": fv(2),
            "salary_gbp": fv(130000),
            "industry": fv("music"),
            "role_substance": fv("leadership"),
        },
        rules={
            "no_agency": RuleVerdict(verdict="pass", confidence=0.9),
            "no_blocked_stack": RuleVerdict(verdict="pass", confidence=0.9),
        },
    )


def test_all_pass_is_apply():
    a = assess(make_cfg(), make_listing(), clean_extraction())
    assert a.decision == "apply"
    assert a.tier_label == "T1"
    assert a.priority == 0
    assert a.fit_score == 100
    assert a.flags == []


def test_any_hard_fail_is_no():
    e = clean_extraction()
    e.features["office_days_per_week"] = fv(5)
    a = assess(make_cfg(), make_listing(), e)
    assert a.decision == "no"
    assert a.priority is None
    assert "gate-fail-office" in a.flags
    assert a.fit_score <= 15  # confirmed dealbreaker band


def test_hard_fail_is_non_compensatory():
    # A perfect T1 role with one confirmed dealbreaker must still be "no".
    e = clean_extraction()
    e.rules["no_agency"] = RuleVerdict(verdict="fail", confidence=0.95)
    a = assess(make_cfg(), make_listing(title="VP of Engineering"), e)
    assert a.decision == "no"


def test_unknown_with_pass_policy_applies_silently():
    e = clean_extraction()
    del e.features["salary_gbp"]  # on_unknown = pass
    a = assess(make_cfg(), make_listing(), e)
    assert a.decision == "apply"
    assert "gate-unknown-salary_floor" not in a.flags
    assert a.fit_score == 98  # one unknown * penalty 2


def test_unknown_with_pass_flag_policy_flags():
    e = clean_extraction()
    del e.features["office_days_per_week"]  # on_unknown = pass_flag
    a = assess(make_cfg(), make_listing(), e)
    assert a.decision == "apply"
    assert "gate-unknown-office" in a.flags


def test_unknown_with_fail_policy_rejects_softly():
    cfg = make_cfg()
    cfg.gates[1].on_unknown = "fail"  # salary_floor becomes reject-on-unknown
    e = clean_extraction()
    del e.features["salary_gbp"]
    a = assess(cfg, make_listing(), e)
    assert a.decision == "no"
    assert "gate-fail-salary_floor" in a.flags
    assert 30 <= a.fit_score <= 69  # soft reject band, not the dealbreaker band


def test_unknown_penalty_demotes_within_tier():
    complete = assess(make_cfg(), make_listing(), clean_extraction())
    e = clean_extraction()
    del e.features["office_days_per_week"]
    del e.features["salary_gbp"]
    uncertain = assess(make_cfg(), make_listing(), e)
    assert uncertain.decision == "apply"
    assert uncertain.tier_label == complete.tier_label == "T1"
    assert uncertain.priority > complete.priority  # demoted within the same bucket
    assert uncertain.priority < 100  # but still ahead of every T2 listing


# --- tier assignment -------------------------------------------------------------

def test_tier_from_title():
    cfg = make_cfg()
    assert assign_tier("Senior Engineering Manager - Billing", extraction(), cfg.priority) \
        == ("T1", 0)
    assert assign_tier("Founding Engineer (AI startup)", extraction(), cfg.priority) == ("T2", 1)
    assert assign_tier("Engineering Manager, UK", extraction(), cfg.priority) == ("T3", 2)
    assert assign_tier("Staff Engineer - London", extraction(), cfg.priority) == ("T4", 3)


def test_substance_beats_title():
    # Staff title, founding substance -> founding tier.
    e = extraction(features={"role_substance": fv("founding", confidence=0.9)})
    assert assign_tier("Staff Software Engineer, New Products", e, make_cfg().priority) == ("T2", 1)


def test_low_confidence_substance_cannot_move_tier():
    e = extraction(features={"role_substance": fv("founding", confidence=0.4)})
    assert assign_tier("Staff Engineer", e, make_cfg().priority) == ("T4", 3)


def test_substance_places_unmatched_title():
    e = extraction(features={"role_substance": fv("leadership", confidence=0.9)})
    assert assign_tier("Head of Product Engineering", e, make_cfg().priority) == ("T1", 0)


def test_no_match_is_none_tier():
    # clean_extraction says leadership substance, so strip it:
    e = clean_extraction()
    del e.features["role_substance"]
    a = assess(make_cfg(), make_listing(title="Quantitative Researcher"), e)
    assert a.decision == "apply"
    assert a.tier_label == "none"
    assert a.priority == 400  # behind every named tier
    assert a.fit_score >= 70  # apply always stays shortlist-visible


def test_tier_ordering_maps_to_fit_bands():
    cfg = make_cfg()
    fits = []
    titles = ["VP of Engineering", "Founding Engineer", "Engineering Manager", "Staff Engineer"]
    for title in titles:
        e = clean_extraction()
        del e.features["role_substance"]
        fits.append(assess(cfg, make_listing(title=title), e).fit_score)
    assert fits == sorted(fits, reverse=True)
    assert all(f >= 70 for f in fits)


# --- extraction plumbing ------------------------------------------------------

def test_build_extract_system_lists_features_and_rules():
    system = build_extract_system(make_cfg().gates)
    assert "office_days_per_week" in system
    assert "no_agency: Reject only if pure consulting/agency work." in system
    assert "no_blocked_stack:" in system
    assert "primary_backend_language" not in system


def test_parse_extraction_tolerates_junk():
    e = parse_extraction({
        "features": {"salary_gbp": {"value": 120000, "confidence": 1.0, "evidence": "£120k"},
                     "bogus": "not-a-dict"},
        "rules": {"no_agency": {"verdict": "pass", "confidence": 0.8}},
        "summary": "s",
    })
    assert e.features["salary_gbp"].value == 120000
    assert "bogus" not in e.features
    assert e.rules["no_agency"].verdict == "pass"


# --- GatedScorer end-to-end (mocked client) -------------------------------------

def mock_extract_response(payload: dict, usage=None) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    # Explicit so getattr(response, "usage") isn't an auto-generated MagicMock that
    # would be bound into sqlite; None => _usage_dict yields nulls.
    resp.usage = usage
    return resp


@patch("jobscout.llm.openai.OpenAI")
def test_gated_scorer_score(mock_openai_cls):
    client = MagicMock()
    mock_openai_cls.return_value = client
    client.chat.completions.create.return_value = mock_extract_response({
        "features": {
            "office_days_per_week": {"value": 2, "confidence": 1.0, "evidence": "2 days"},
            "salary_gbp": {"value": 140000, "confidence": 1.0, "evidence": "£120–140k"},
            "industry": {"value": "music", "confidence": 0.9, "evidence": "music startup"},
            "role_substance": {"value": "leadership", "confidence": 0.9,
                               "evidence": "lead the team"},
        },
        "rules": {
            "no_agency": {"verdict": "pass", "confidence": 0.9, "evidence": "own product"},
            "no_blocked_stack": {"verdict": "pass", "confidence": 0.9, "evidence": "no hard stack req"},
        },
        "summary": "Head of Engineering at a music startup.",
    })

    score = GatedScorer(make_cfg()).score(make_listing())

    assert score.decision == "apply"
    assert score.tier_label == "T1"
    assert score.priority is not None
    assert score.fit_score >= 70  # derived compat: apply <=> fit >= 70
    assert score.tier == "bulk"
    assert len(score.gate_results) == 5
    # no_blocked_stack passed; office unknown flagged but on_unknown=pass_flag
    assert score.fit_score == 100


@patch("jobscout.llm.openai.OpenAI")
def test_gated_scorer_rejects_below_70(mock_openai_cls):
    client = MagicMock()
    mock_openai_cls.return_value = client
    client.chat.completions.create.return_value = mock_extract_response({
        "features": {"office_days_per_week": {"value": 5, "confidence": 1.0, "evidence": "onsite"}},
        "rules": {},
        "summary": "Fully in-office role.",
    })

    score = GatedScorer(make_cfg()).score(make_listing())
    assert score.decision == "no"
    assert score.fit_score < 70


# --- producer selection ---------------------------------------------------------

@patch("jobscout.llm.openai.OpenAI")
def test_build_scorer_selects_by_config(mock_openai_cls):
    mock_openai_cls.return_value = MagicMock()
    profile = MagicMock()
    profile.summary, profile.skills, profile.seniority = "s", [], ""
    profile.domains, profile.must_haves, profile.nice_to_haves = [], [], []
    profile.dealbreakers, profile.raw_goals = [], ""

    assert isinstance(build_scorer(make_cfg(), profile), GatedScorer)

    additive = make_cfg()
    additive.ai = AIConfig(api_keys={"openrouter": "k"}, bulk_model="test/model", scorer="additive")
    assert isinstance(build_scorer(additive, profile), BulkScorer)


# --- consensus / self-consistency ------------------------------------------------

def _a(decision, tier_label="none", fit_score=70, priority=None) -> Assessment:
    return Assessment(decision=decision, tier_label=tier_label, fit_score=fit_score,
                      priority=priority, rationale="r", flags=[], gate_results=[])


def test_consensus_single_returns_itself():
    a = _a("apply", "T1", 95, 0)
    assert consensus_assessment([a]) is a


def test_consensus_majority_apply_wins():
    # apply / no / apply -> apply (the flip case that self-consistency resolves)
    out = consensus_assessment([_a("apply", "T2", 90, 100), _a("no", "none", 15),
                                _a("apply", "T2", 90, 100)])
    assert out.decision == "apply"
    assert out.tier_label == "T2"


def test_consensus_majority_no_wins():
    out = consensus_assessment([_a("no", "none", 15), _a("no", "none", 15),
                                _a("apply", "T4", 70, 300)])
    assert out.decision == "no"


def test_consensus_representative_from_winning_side_only():
    # the lone "no" must not supply tier/fit when "apply" wins
    out = consensus_assessment([_a("apply", "T1", 92, 8), _a("apply", "T1", 92, 8),
                                _a("no", "none", 5)])
    assert out.decision == "apply"
    assert out.tier_label == "T1"
    assert out.fit_score == 92


def test_consensus_split_is_flagged():
    out = consensus_assessment([_a("apply", "T2", 90, 100), _a("apply", "T2", 90, 100),
                                _a("no", "none", 15)])
    assert any(f.startswith("consensus-split-2of3") for f in out.flags)
    assert out.rationale.startswith("[consensus 2/3]")


def test_consensus_unanimous_not_flagged():
    out = consensus_assessment([_a("apply", "T1", 95, 0), _a("apply", "T1", 95, 0),
                                _a("apply", "T1", 95, 0)])
    assert not any("consensus-split" in f for f in out.flags)
    assert not out.rationale.startswith("[consensus")


def test_consensus_locked_two_agreeing_with_one_left():
    # decision AND tier agree, 1 repeat outstanding -> the third can't change it.
    votes = [_a("apply", "T1"), _a("apply", "T1")]
    assert consensus_locked(votes, remaining=1) is True


def test_consensus_locked_single_vote_not_locked_when_repeats_remain():
    # one vote, one outstanding: a future run could still tie/flip it.
    assert consensus_locked([_a("apply", "T1")], remaining=1) is False


def test_consensus_locked_single_vote_locked_when_nothing_remains():
    assert consensus_locked([_a("apply", "T1")], remaining=0) is True


def test_consensus_locked_decision_split_not_locked():
    votes = [_a("apply", "T1"), _a("no", "none")]
    assert consensus_locked(votes, remaining=1) is False


def test_consensus_locked_decision_agrees_but_tier_split_not_locked():
    # both apply (decision locked) but different tiers with a repeat left: tier
    # is not yet settled, so keep voting — matches the verified 2.09-calls rule.
    votes = [_a("apply", "T1"), _a("apply", "T2")]
    assert consensus_locked(votes, remaining=1) is False


def test_consensus_locked_empty_is_false():
    assert consensus_locked([], remaining=2) is False


@patch("jobscout.llm.openai.OpenAI")
def test_gated_scorer_early_stops_when_first_two_agree(mock_openai_cls):
    # Two agreeing votes lock decision+tier, so the third call is skipped — this is
    # the ~30%-cheaper common case, provably identical to a fixed majority-of-3.
    client = MagicMock()
    mock_openai_cls.return_value = client
    client.chat.completions.create.return_value = mock_extract_response({
        "features": {
            "office_days_per_week": {"value": 2, "confidence": 1.0, "evidence": "2 days"},
            "role_substance": {"value": "leadership", "confidence": 0.9, "evidence": "lead"},
        },
        "rules": {"no_agency": {"verdict": "pass", "confidence": 0.9, "evidence": "product"}},
        "summary": "Head of Eng.",
    })
    cfg = make_cfg()
    cfg.ai.scorer_repeats = 3
    score = GatedScorer(cfg).score(make_listing())
    assert client.chat.completions.create.call_count == 2
    assert score.decision == "apply"


@patch("jobscout.assess.GatedScorer.extract")
def test_gated_scorer_runs_mandatory_calls_in_parallel(mock_extract):
    # A barrier of 2 only clears if both mandatory extractions are in flight at once,
    # so this deadlocks (and times out) under a serial implementation — proof the two
    # calls actually overlap.
    import threading

    barrier = threading.Barrier(2, timeout=5)
    good = Extraction(
        features={"role_substance": FeatureValue(value="leadership", confidence=0.9,
                                                 evidence="lead")},
        rules={}, summary="Head of Eng.",
    )

    def extract(listing, rep):
        barrier.wait()
        return good

    mock_extract.side_effect = extract
    cfg = make_cfg()
    cfg.ai.scorer_repeats = 3
    with patch("jobscout.llm.openai.OpenAI"):
        score = GatedScorer(cfg).score(make_listing())
    assert score.decision == "apply"
    assert mock_extract.call_count == 2  # both agreed, so the third call is skipped


@patch("jobscout.llm.openai.OpenAI")
def test_gated_scorer_parallel_batch_is_store_safe(mock_openai_cls, tmp_path):
    # Real Store + real worker threads: the mandatory repeats read/write the sqlite
    # extraction cache concurrently. A thread-unsafe connection would raise here.
    from jobscout.store import Store

    client = MagicMock()
    mock_openai_cls.return_value = client
    client.chat.completions.create.return_value = mock_extract_response({
        "features": {"role_substance": {"value": "leadership", "confidence": 0.9,
                                        "evidence": "lead"}},
        "rules": {"no_agency": {"verdict": "pass", "confidence": 0.9, "evidence": "product"}},
        "summary": "Head of Eng.",
    })
    with Store(tmp_path / "cache.db") as store:
        cfg = make_cfg()
        cfg.ai.scorer_repeats = 3
        scorer = GatedScorer(cfg, store=store)
        listing, _ = store.upsert_listing(make_listing())

        score = scorer.score(listing)
        assert score.decision == "apply"
        assert client.chat.completions.create.call_count == 2  # two agreed, third skipped
        th = content_hash(listing)
        for rep in (0, 1):
            assert store.get_extraction(
                listing.id, cfg.ai.bulk_model, scorer._prompt_hash, rep, th
            ) is not None


@patch("jobscout.assess.GatedScorer.extract")
def test_gated_scorer_runs_third_call_when_first_two_disagree(mock_extract):
    # rep0/rep1 disagree on decision, so nothing is locked and the tiebreak call runs.
    yes = Extraction(
        features={"role_substance": FeatureValue(value="leadership", confidence=0.9,
                                                 evidence="lead")},
        rules={}, summary="Head of Eng.",
    )
    no = Extraction(
        features={"office_days_per_week": FeatureValue(value=5, confidence=1.0,
                                                       evidence="onsite")},
        rules={}, summary="Onsite.",
    )
    mock_extract.side_effect = [yes, no, yes]  # 2:1 apply wins after the third
    cfg = make_cfg()
    cfg.ai.scorer_repeats = 3
    with patch("jobscout.llm.openai.OpenAI"):
        score = GatedScorer(cfg).score(make_listing())
    assert mock_extract.call_count == 3
    assert score.decision == "apply"


@patch("jobscout.assess.GatedScorer.extract")
def test_gated_scorer_repeats_tolerates_partial_failure(mock_extract):
    # one of the three extractions raises; the vote must still stand on the other two
    good = Extraction(
        features={"role_substance": FeatureValue(value="leadership", confidence=0.9,
                                                 evidence="lead")},
        rules={}, summary="Head of Eng.",
    )
    mock_extract.side_effect = [good, RuntimeError("boom"), good]
    cfg = make_cfg()
    cfg.ai.scorer_repeats = 3
    with patch("jobscout.llm.openai.OpenAI"):
        score = GatedScorer(cfg).score(make_listing())
    assert score.decision == "apply"
    assert mock_extract.call_count == 3


@patch("jobscout.assess.GatedScorer.extract")
def test_gated_scorer_repeats_raises_when_all_fail(mock_extract):
    mock_extract.side_effect = [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")]
    cfg = make_cfg()
    cfg.ai.scorer_repeats = 3
    with patch("jobscout.llm.openai.OpenAI"):
        try:
            GatedScorer(cfg).score(make_listing())
            assert False, "should have raised"
        except RuntimeError:
            pass


# --- extraction cache (store-backed) ---------------------------------------------

@patch("jobscout.llm.openai.OpenAI")
def test_gated_scorer_caches_extraction_to_store(mock_openai_cls, tmp_path):
    from jobscout.store import Store

    client = MagicMock()
    mock_openai_cls.return_value = client
    client.chat.completions.create.return_value = mock_extract_response({
        "features": {"role_substance": {"value": "leadership", "confidence": 0.9,
                                        "evidence": "lead"}},
        "rules": {"no_agency": {"verdict": "pass", "confidence": 0.9, "evidence": "product"}},
        "summary": "Head of Eng.",
    })

    with Store(tmp_path / "cache.db") as store:
        cfg = make_cfg()
        scorer = GatedScorer(cfg, store=store)
        listing, _ = store.upsert_listing(make_listing())

        scorer.score(listing)
        assert client.chat.completions.create.call_count == 1

        cached = store.get_extraction(
            listing.id, cfg.ai.bulk_model, scorer._prompt_hash, 0, content_hash(listing)
        )
        assert cached is not None

        # A second score() run must hit the cache instead of calling the LLM again.
        scorer.score(listing)
        assert client.chat.completions.create.call_count == 1


@patch("jobscout.llm.openai.OpenAI")
def test_enriched_description_invalidates_cached_extraction(mock_openai_cls, tmp_path):
    # The bug this fixes: a fuller JD must NOT reuse the thin-JD extraction.
    from jobscout.store import Store

    client = MagicMock()
    mock_openai_cls.return_value = client
    client.chat.completions.create.return_value = mock_extract_response({
        "features": {"role_substance": {"value": "leadership", "confidence": 0.9,
                                        "evidence": "lead"}},
        "rules": {"no_agency": {"verdict": "pass", "confidence": 0.9, "evidence": "product"}},
        "summary": "Head of Eng.",
    })
    with Store(tmp_path / "cache.db") as store:
        cfg = make_cfg()
        cfg.ai.scorer_repeats = 1
        scorer = GatedScorer(cfg, store=store)
        listing, _ = store.upsert_listing(make_listing(description="thin JD"))

        scorer.score(listing)
        assert client.chat.completions.create.call_count == 1

        # Same id, richer description -> different content hash -> must re-extract.
        enriched = listing.model_copy(update={"description": "a much fuller job description"})
        scorer.score(enriched)
        assert client.chat.completions.create.call_count == 2

        # Re-scoring the enriched text again now hits the cache (no third call).
        scorer.score(enriched)
        assert client.chat.completions.create.call_count == 2


@patch("jobscout.llm.openai.OpenAI")
def test_gated_scorer_persists_usage(mock_openai_cls, tmp_path):
    from types import SimpleNamespace

    from jobscout.store import Store

    client = MagicMock()
    mock_openai_cls.return_value = client
    usage = SimpleNamespace(cost=0.00042, prompt_tokens=1500, completion_tokens=300)
    client.chat.completions.create.return_value = mock_extract_response(
        {"features": {"role_substance": {"value": "leadership", "confidence": 0.9,
                                         "evidence": "lead"}},
         "rules": {"no_agency": {"verdict": "pass", "confidence": 0.9, "evidence": "product"}},
         "summary": "Head of Eng."},
        usage=usage,
    )
    with Store(tmp_path / "cache.db") as store:
        cfg = make_cfg()
        cfg.ai.scorer_repeats = 1
        scorer = GatedScorer(cfg, store=store)
        listing, _ = store.upsert_listing(make_listing())
        scorer.score(listing)

        row = store.conn.execute(
            "SELECT cost, prompt_tokens, completion_tokens FROM extractions "
            "WHERE listing_id = ?", (listing.id,)
        ).fetchone()
        assert row["cost"] == 0.00042
        assert row["prompt_tokens"] == 1500
        assert row["completion_tokens"] == 300


@patch("jobscout.llm.openai.OpenAI")
def test_gated_scorer_without_store_never_caches(mock_openai_cls, tmp_path):
    from jobscout.store import Store

    client = MagicMock()
    mock_openai_cls.return_value = client
    client.chat.completions.create.return_value = mock_extract_response({
        "features": {}, "rules": {}, "summary": "s",
    })

    cfg = make_cfg()
    listing = make_listing()
    GatedScorer(cfg).score(listing)  # no store passed
    GatedScorer(cfg).score(listing)
    assert client.chat.completions.create.call_count == 2

    with Store(tmp_path / "empty.db") as store:
        assert store.get_extraction(listing.id, cfg.ai.bulk_model, "anything", 0) is None
