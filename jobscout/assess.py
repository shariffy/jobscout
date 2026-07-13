"""Gated + tiered assessment: the LLM does perception, Python does policy.

Stage 1 (LLM, one call per listing): extract the canonical features (features.py)
and evaluate each natural-language gate rule — every answer with confidence and a
supporting quote. Narrow factual outputs are far lower-variance than a 0-100 vibe.

Stage 2 (pure Python): evaluate the structured [[gates]] predicates against the
features, combine with the NL verdicts, and apply each gate's on_unknown policy.
Only a high-confidence detection can hard-fail a gate; anything uncertain is
"unknown" and handled by policy. Any effective fail => decision = "no".

Stage 3 (pure Python): for the apply set, assign a priority tier from the title
and the extracted role substance (substance beats title), then rank
bucketed-lexicographically: tier is the primary key, per-unknown-gate penalties
break ties within it. A derived fit_score is emitted only as migration glue —
apply lands in 70-100 by tier, no lands below 70 by severity — so the existing
fit >= 70 shortlist plumbing keeps working until it is dropped.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .config import Config, assessment_config_hash
from .features import FEATURES
from .gates import GateConfig, PriorityConfig
from .llm import openrouter_client, with_retries
from .models import CandidateProfile, Listing, Score
from .scoring import _listing_text, _parse_score
from .store import Store

# role_substance below this confidence cannot move a listing between tiers.
SUBSTANCE_MIN_CONFIDENCE = 0.7

_APPLY_FLOOR = 70  # fit >= 70 must stay equivalent to decision == "apply"


class FeatureValue(BaseModel):
    value: Any = None
    confidence: float = 0.0
    evidence: str = ""


class RuleVerdict(BaseModel):
    verdict: str = "unknown"  # pass | fail | unknown
    confidence: float = 0.0
    evidence: str = ""


class Extraction(BaseModel):
    """Everything Stage 1 returns for one listing."""

    features: dict[str, FeatureValue] = Field(default_factory=dict)
    rules: dict[str, RuleVerdict] = Field(default_factory=dict)
    summary: str = ""


class GateResult(BaseModel):
    name: str
    status: str  # pass | fail | unknown
    reason: str = ""
    evidence: str = ""
    confidence: float = 0.0


class Assessment(BaseModel):
    decision: str  # apply | no
    priority: int | None = None
    tier_label: str = "none"
    fit_score: int = 0
    rationale: str = ""
    flags: list[str] = Field(default_factory=list)
    gate_results: list[GateResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1 — extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_HEADER = """\
You are a precise job-listing information extractor. Read the job listing and return
ONLY a JSON object — no markdown, no prose — exactly:
{
  "features": {"<name>": {"value": <value or null>, "confidence": <0.0-1.0>, "evidence": "<short quote or ''>"}, ...},
  "rules": {"<name>": {"verdict": "pass" | "fail" | "unknown", "confidence": <0.0-1.0>, "evidence": "<short quote or ''>"}, ...},
  "summary": "<one factual sentence: what the role actually is, at what kind of company>"
}
Include one entry per feature and per rule listed below, and nothing else.

Ground rules:
- value = null whenever the listing does not state or clearly imply the fact. NEVER guess.
- confidence: 1.0 = the listing states it explicitly; ~0.6 = a reasonable inference from
  clear signals; below 0.5 = weak — prefer null/unknown instead.
- evidence: the shortest quote from the listing that supports the value or verdict.
- The Location field is scraped from a job card and may be wrong; prefer the office/remote
  policy stated in the description when they conflict.

Features to extract:
"""

_RULES_HEADER = """
Rules to evaluate — each states a REJECT condition about the listing itself:
verdict "fail" = the reject condition is clearly confirmed; "pass" = clearly not the
case; "unknown" = the listing does not say. Judge only from the listing text.
"""


def build_extract_system(gates: list[GateConfig]) -> str:
    lines = [_EXTRACT_HEADER]
    for name, desc in FEATURES.items():
        lines.append(f"- {name}: {desc}")
    nl_gates = [g for g in gates if g.rule]
    if nl_gates:
        lines.append(_RULES_HEADER)
        for g in nl_gates:
            lines.append(f"- {g.name}: {g.rule}")
    else:
        lines.append('\nNo rules to evaluate — return "rules": {}.')
    return "\n".join(lines)


def extract_prompt_hash(system: str, reasoning: dict[str, Any] | None) -> str:
    """Cache key for a Stage-1 extraction call, independent of listing/repeat.

    Folds reasoning in alongside the prompt text so flipping ai.bulk_reasoning
    (e.g. off -> low effort) invalidates cached extractions instead of serving a
    value produced under a different thinking budget.
    """
    payload = json.dumps({"system": system, "reasoning": reasoning}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def parse_extraction(data: dict) -> Extraction:
    features = {
        name: FeatureValue.model_validate(v)
        for name, v in (data.get("features") or {}).items()
        if isinstance(v, dict)
    }
    rules = {
        name: RuleVerdict.model_validate(v)
        for name, v in (data.get("rules") or {}).items()
        if isinstance(v, dict)
    }
    return Extraction(features=features, rules=rules, summary=str(data.get("summary") or ""))


# ---------------------------------------------------------------------------
# Stage 2 — gate engine (pure Python)
# ---------------------------------------------------------------------------

def _norm(v: Any) -> Any:
    return v.strip().casefold() if isinstance(v, str) else v


def _eval_op(op: str, actual: Any, expected: Any) -> bool:
    """Evaluate one structured predicate. Raises ValueError when the value can't
    be compared (caller treats that as unknown)."""
    try:
        if op in ("<=", ">="):
            a, e = float(actual), float(expected)
            return a <= e if op == "<=" else a >= e
        if op in ("==", "!="):
            try:
                eq = float(actual) == float(expected)
            except (TypeError, ValueError):
                eq = _norm(str(actual)) == _norm(str(expected))
            return eq if op == "==" else not eq
        if op in ("in", "not_in"):
            members = [_norm(str(e)) for e in (expected or [])]
            hit = _norm(str(actual)) in members
            return hit if op == "in" else not hit
        if op == "contains":
            if isinstance(actual, list):
                return _norm(str(expected)) in [_norm(str(a)) for a in actual]
            return _norm(str(expected)) in _norm(str(actual))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot compare {actual!r} {op} {expected!r}") from exc
    raise ValueError(f"unknown op {op!r}")


def evaluate_gate(
    gate: GateConfig,
    features: dict[str, FeatureValue],
    rules: dict[str, RuleVerdict],
) -> GateResult:
    if gate.rule:
        v = rules.get(gate.name)
        if v is None or v.verdict not in ("pass", "fail"):
            return GateResult(name=gate.name, status="unknown", reason="no verdict from listing")
        if v.verdict == "fail" and v.confidence < gate.min_confidence:
            # Low-confidence detections must not hard-fail — the false-dealbreaker fix.
            return GateResult(
                name=gate.name, status="unknown",
                reason=f"fail verdict below confidence {gate.min_confidence}",
                evidence=v.evidence, confidence=v.confidence,
            )
        return GateResult(
            name=gate.name, status=v.verdict,
            reason="rule " + ("confirmed" if v.verdict == "fail" else "not triggered"),
            evidence=v.evidence, confidence=v.confidence,
        )

    fv = features.get(gate.feature)
    if fv is None or fv.value is None:
        return GateResult(name=gate.name, status="unknown", reason=f"{gate.feature} not stated")
    if fv.confidence < gate.min_confidence:
        return GateResult(
            name=gate.name, status="unknown",
            reason=f"{gate.feature}={fv.value!r} below confidence {gate.min_confidence}",
            evidence=fv.evidence, confidence=fv.confidence,
        )
    try:
        ok = _eval_op(gate.op, fv.value, gate.value)
    except ValueError as exc:
        return GateResult(
            name=gate.name, status="unknown", reason=str(exc),
            evidence=fv.evidence, confidence=fv.confidence,
        )
    return GateResult(
        name=gate.name,
        status="pass" if ok else "fail",
        reason=f"{gate.feature}={fv.value!r} {gate.op} {gate.value!r}",
        evidence=fv.evidence,
        confidence=fv.confidence,
    )


def evaluate_gates(
    gates: list[GateConfig],
    extraction: Extraction,
) -> list[GateResult]:
    return [evaluate_gate(g, extraction.features, extraction.rules) for g in gates]


# ---------------------------------------------------------------------------
# Stage 3 — priority engine (pure Python)
# ---------------------------------------------------------------------------

def _norm_title(s: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def assign_tier(
    listing_title: str,
    extraction: Extraction,
    priority_cfg: PriorityConfig,
) -> tuple[str, int | None]:
    """Return (tier name, tier index). Substance beats title: the best (earliest)
    tier reached by either route wins, so a Staff-titled founding role lands in
    the founding tier. ("none", None) when nothing matches."""
    title = _norm_title(listing_title)
    substance = None
    fv = extraction.features.get("role_substance")
    if fv and isinstance(fv.value, str) and fv.confidence >= SUBSTANCE_MIN_CONFIDENCE:
        substance = _norm(fv.value)

    best: int | None = None
    for idx, tier in enumerate(priority_cfg.tiers):
        by_title = any(_norm_title(t) in title for t in tier.titles)
        by_substance = substance is not None and substance in [_norm(s) for s in tier.substances]
        if by_title or by_substance:
            best = idx
            break
    if best is None:
        return "none", None
    return priority_cfg.tiers[best].name, best


def _fit_band(tier_idx: int | None) -> tuple[int, int]:
    """(floor, top) of the derived apply-band fit for a tier index; the top tier
    reaches 100 and each tier steps down, never below the apply floor."""
    if tier_idx is None:
        return _APPLY_FLOOR, _APPLY_FLOOR + 2
    top = max(_APPLY_FLOOR + 2, 100 - 8 * tier_idx)
    return max(_APPLY_FLOOR, top - 8), top


def assess(cfg: Config, listing: Listing, extraction: Extraction) -> Assessment:
    """Stages 2 + 3: turn one extraction into a decision + priority (+ derived fit)."""
    results = evaluate_gates(cfg.gates, extraction)
    by_name = {g.name: g for g in cfg.gates}

    confirmed = [r for r in results if r.status == "fail"]
    policy_fails = [
        r for r in results
        if r.status == "unknown" and by_name[r.name].on_unknown == "fail"
    ]
    unknowns = [
        r for r in results
        if r.status == "unknown" and by_name[r.name].on_unknown != "fail"
    ]

    flags = [f"gate-fail-{r.name}" for r in confirmed + policy_fails]
    flags += [
        f"gate-unknown-{r.name}" for r in unknowns
        if by_name[r.name].on_unknown == "pass_flag"
    ]

    if confirmed or policy_fails:
        # Reject band: confirmed dealbreakers sit at the bottom; rejections that
        # exist only because unknowns default to fail land in the soft band.
        if confirmed:
            fit = max(5, 15 - 5 * (len(confirmed) - 1))
        else:
            fit = max(30, 69 - 10 * len(policy_fails))
        failed_bits = "; ".join(f"{r.name} ({r.reason})" for r in confirmed + policy_fails)
        rationale = f"{extraction.summary} NO — failed gates: {failed_bits}."
        return Assessment(
            decision="no", priority=None, tier_label="none", fit_score=fit,
            rationale=rationale.strip(), flags=flags, gate_results=results,
        )

    tier_label, tier_idx = assign_tier(listing.title, extraction, cfg.priority)
    penalty = cfg.priority.unknown_penalty * len(unknowns)
    # Bucketed-lexicographic rank: tier is the primary key (a whole bucket per
    # tier), uncertainty penalties only reorder within the bucket.
    bucket = tier_idx if tier_idx is not None else len(cfg.priority.tiers)
    priority = bucket * 100 + penalty

    floor, top = _fit_band(tier_idx)
    fit = max(floor, top - penalty)

    bits = [f"APPLY {tier_label}" if tier_label != "none" else "APPLY (no tier match)"]
    if unknowns:
        bits.append("unknown: " + ", ".join(r.name for r in unknowns))
    rationale = f"{extraction.summary} {' — '.join(bits)}."
    return Assessment(
        decision="apply", priority=priority, tier_label=tier_label, fit_score=fit,
        rationale=rationale.strip(), flags=flags, gate_results=results,
    )


def consensus_assessment(assessments: list[Assessment]) -> Assessment:
    """Self-consistency vote over repeated assessments of one listing.

    Returns the majority decision; among the runs that agree with it, picks a stable
    representative (the modal tier, then the median fit) so priority/tier/fit come from
    a run that actually reached the winning decision. This turns a cheaper, less
    deterministic extractor into a stable decider — a per-listing flip needs the
    minority to become the majority, which K independent draws make far rarer.
    """
    from collections import Counter

    if len(assessments) == 1:
        return assessments[0]
    decisions = [a.decision for a in assessments]
    majority = Counter(decisions).most_common(1)[0][0]
    winners = [a for a in assessments if a.decision == majority]
    modal_tier = Counter(a.tier_label for a in winners).most_common(1)[0][0]
    cands = [a for a in winners if a.tier_label == modal_tier] or winners
    cands = sorted(cands, key=lambda a: a.fit_score)
    rep = cands[len(cands) // 2]
    n, k = len(winners), len(assessments)
    if n < k:  # note the split so a shaky verdict is visible downstream
        rep = rep.model_copy(update={
            "rationale": f"[consensus {n}/{k}] {rep.rationale}",
            "flags": [*rep.flags, f"consensus-split-{n}of{k}"],
        })
    return rep


def consensus_locked(assessments: list[Assessment], remaining: int) -> bool:
    """True when `remaining` more assessments cannot change the majority decision
    or the modal tier among the decision-winners — so voting can stop early.

    This makes early-stopping exactly equivalent to a fixed K-way majority on the
    two fields that drive the pipeline (decision, then tier): once a decision leads
    by more than the votes still outstanding, no future run can catch it, so the
    remaining calls are pure cost with no effect on the outcome. Only the derived
    representative fields (fit/priority) can still shift — the same latitude
    consensus_assessment already takes when it picks a median representative.

    For K=3 this drops the third call whenever the first two agree on decision+tier.
    """
    from collections import Counter

    if not assessments:
        return False
    dec_counts = Counter(a.decision for a in assessments)
    top_dec, top_n = dec_counts.most_common(1)[0]
    # An unseen decision could still take up to `remaining` future votes, so the
    # leader is only locked when it strictly clears the best rival PLUS those votes.
    best_rival = max((c for d, c in dec_counts.items() if d != top_dec), default=0)
    if top_n <= best_rival + remaining:
        return False
    winners = [a for a in assessments if a.decision == top_dec]
    tier_counts = Counter(a.tier_label for a in winners)
    top_tier, top_tn = tier_counts.most_common(1)[0]
    best_other_tier = max((c for t, c in tier_counts.items() if t != top_tier), default=0)
    return top_tn > best_other_tier + remaining


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class GatedScorer:
    """Drop-in Score producer: one extraction call, then pure-Python policy.

    The candidate profile is not consulted — every user-specific rule lives in
    config ([[gates]] / [priority]); it is accepted only for BulkScorer signature
    parity."""

    def __init__(
        self, cfg: Config, profile: CandidateProfile | None = None, store: Store | None = None
    ) -> None:
        self._cfg = cfg
        self._client = openrouter_client(cfg)
        self._system = build_extract_system(cfg.gates)
        self._reasoning = cfg.ai.bulk_reasoning
        self._repeats = max(1, cfg.ai.scorer_repeats)
        self._version = assessment_config_hash(cfg)
        # Stage-1 cache: a listing/model/prompt/repeat already extracted skips the
        # LLM call entirely, so gate/priority-only config edits (which don't change
        # extract_prompt_hash) never re-pay for extraction on rescore.
        self._store = store
        self._prompt_hash = extract_prompt_hash(self._system, self._reasoning)

    def extract(self, listing: Listing, repeat_idx: int = 0) -> Extraction:
        if self._store is not None and listing.id is not None:
            cached = self._store.get_extraction(
                listing.id, self._cfg.ai.bulk_model, self._prompt_hash, repeat_idx
            )
            if cached is not None:
                return Extraction.model_validate_json(cached)

        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": f"Extract from this listing:\n\n{_listing_text(listing)}"},
        ]
        extra_body: dict = {"usage": {"include": True}}
        if self._reasoning is not None:
            extra_body["reasoning"] = self._reasoning

        last_exc: Exception | None = None
        raw = ""
        for attempt in range(2):
            response = with_retries(
                lambda: self._client.chat.completions.create(
                    model=self._cfg.ai.bulk_model,
                    messages=messages,
                    max_tokens=4096,
                    extra_body=extra_body,
                ),
                label=self._cfg.ai.bulk_model,
            )
            raw = (response.choices[0].message.content or "") if response.choices else ""
            try:
                extraction = parse_extraction(_parse_score(raw))
                if self._store is not None and listing.id is not None:
                    self._store.save_extraction(
                        listing.id, self._cfg.ai.bulk_model, self._prompt_hash, repeat_idx,
                        extraction.model_dump_json(),
                    )
                return extraction
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user",
                         "content": "Reply with only the JSON object, no other text."},
                    ]
        raise ValueError(
            f"Could not parse extraction after 2 attempts. Last raw: {raw!r}"
        ) from last_exc

    def _extract_and_assess(self, listing: Listing, rep: int) -> Assessment:
        return assess(self._cfg, listing, self.extract(listing, rep))

    def score(self, listing: Listing) -> Score:
        # Self-consistency vote with early stop and parallelism.
        #
        # A K-way majority can never be settled by fewer than floor(K/2)+1 agreeing
        # runs, so that many repeats are ALWAYS needed — run them concurrently, since
        # the extraction calls are independent (the vote's worker threads reach the
        # extraction cache through Store's lock). The rest are conditional: run them
        # serially, stopping the instant decision+tier lock (see consensus_locked), so
        # the early-stop savings hold exactly — parallelism cuts latency, not calls.
        # For K=3 that is two concurrent calls, then a third only when they disagree.
        #
        # Tolerate partial failure — the vote still stands if a repeat errors out;
        # only give up if every attempt fails.
        k = self._repeats
        mandatory = k // 2 + 1
        assessments: list[Assessment] = []
        last_exc: Exception | None = None

        if mandatory == 1:
            try:
                assessments.append(self._extract_and_assess(listing, 0))
            except Exception as exc:
                last_exc = exc
        else:
            with ThreadPoolExecutor(max_workers=mandatory) as pool:
                futures = [pool.submit(self._extract_and_assess, listing, r)
                           for r in range(mandatory)]
                for fut in futures:  # collect in rep order for a deterministic vote
                    try:
                        assessments.append(fut.result())
                    except Exception as exc:
                        last_exc = exc

        for rep in range(mandatory, k):
            if consensus_locked(assessments, k - rep):
                break
            try:
                assessments.append(self._extract_and_assess(listing, rep))
            except Exception as exc:
                last_exc = exc

        if not assessments:
            raise last_exc if last_exc else ValueError("extraction failed")
        a = consensus_assessment(assessments)
        return Score(
            listing_id=listing.id,
            fit_score=a.fit_score,
            rationale=a.rationale,
            flags=a.flags,
            breakdown={},
            model=self._cfg.ai.bulk_model,
            tier="bulk",
            decision=a.decision,
            priority=a.priority,
            tier_label=a.tier_label,
            gate_results=[g.model_dump() for g in a.gate_results],
            assessment_version=self._version,
            scored_at=datetime.now(UTC),
        )

    def score_batch(self, listings: list[Listing], progress_cb=None) -> list[Score]:
        scores = []
        for i, listing in enumerate(listings):
            score = self.score(listing)
            scores.append(score)
            if progress_cb:
                progress_cb(i + 1, len(listings), listing, score)
        return scores


def build_scorer(cfg: Config, profile: CandidateProfile, store: Store | None = None):
    """Select the Score producer from ai.scorer ("additive" | "gated").

    store is optional and only consulted by GatedScorer, to cache Stage-1
    extractions (see GatedScorer.extract); pass the open Store so a rescore reuses
    already-paid-for extractions.
    """
    if cfg.ai.scorer == "gated":
        return GatedScorer(cfg, profile, store=store)
    from .scoring import BulkScorer

    return BulkScorer(cfg, profile)
