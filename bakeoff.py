#!/usr/bin/env python3
"""Scoring model bake-off — with a Tier-1 label-free scorecard.

Re-scores a stratified sample of listings across candidate models and reports,
per model: the fit_score it gives each listing (with rationale, so you can eyeball
against your own judgement), how far it drifts from your current stored scores,
and the token cost — both for the sample and projected to the whole corpus.

On top of that it runs a *label-free* Tier-1 scorecard, aimed at screening a cost
challenger (e.g. DeepSeek) for a disqualifying flaw rather than proving accuracy:

  Tier-1.1  Rubric-internal consistency — pure post-processors on the JSON each
            model already returned (no gold label): does fit_score equal
            50 + Σ(breakdown)? is every term inside its allowed range? do the
            flags cohere with the numbers (a flagged dealbreaker must be penalised)?
            A model that contradicts its own rubric is making provable errors.
  Tier-1.3  Test–retest reliability — with --repeats N, score each listing N times
            and report the per-listing score variance. Same input, wildly different
            output is a trust failure you can measure without any labels.

Run from the repo root (reuses the jobscout package and its config/profile/store):

    .venv/bin/python bakeoff.py                 # OpenRouter arm (uses config.toml key)
    .venv/bin/python bakeoff.py --smoke         # 1 free-ish call to check the key/balance
    .venv/bin/python bakeoff.py --n 18          # sample size (spread across score buckets)
    .venv/bin/python bakeoff.py --ids 12 55 90  # score specific listing ids
    .venv/bin/python bakeoff.py --repeats 4     # 4× per listing for test-retest σ

To test the OpenAI arm you need `pip install openai` and OPENAI_API_KEY set; then:

    .venv/bin/python bakeoff.py --openai
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import openai

from jobscout.config import Config, load_config
from jobscout.models import Listing
from jobscout.profile import build_profile
from jobscout.scoring import _build_system, _listing_text, _parse_score
from jobscout.store import Store

# jobscout/llm.py now points at Requesty; this bake-off's "openrouter arm" needs
# OpenRouter specifically, so it keeps its own client pinned there (same pattern
# as scorecard.py / scorecard_gated.py).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_client(cfg: Config) -> openai.OpenAI:
    key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("No OpenRouter API key — set OPEN_ROUTER_API_KEY (or OPENROUTER_API_KEY).")
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

# ---------------------------------------------------------------------------
# Price table — $ per 1M tokens (input, output). As of July 2026. Sources noted
# in the conversation; adjust if pricing moves. Sonnet 5 sticker shown; intro
# pricing is $2/$10 through 2026-08-31.
# ---------------------------------------------------------------------------
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.6-terra": (2.5, 15.0),
    "gpt-5.6-luna": (1.0, 6.0),
    "gpt-5.6-sol": (5.0, 30.0),
}

OPENAI_MODELS_DEFAULT = ["gpt-5.6-terra", "gpt-5.6-luna"]

# OpenRouter models to run, each with its reasoning config (None = provider default,
# {"effort": "low"} sets effort, {"enabled": False} turns reasoning off).
# NOTE: the three non-":free" ids are best-guess canonical slugs — if one 404s the
# run shows it as an error column and the id is a one-line fix here.
OPENROUTER_MODELS: list[tuple[str, dict | None]] = [
    ("anthropic/claude-sonnet-5", {"effort": "low"}),          # Sonnet 5, low reasoning
    ("google/gemini-3.5-flash", None),                         # Gemini 3.5 Flash, default
    ("deepseek/deepseek-v4-pro", {"enabled": False}),          # DeepSeek v4 Pro, no reasoning
    ("nvidia/nemotron-3-super-120b-a12b:free", None),
    ("qwen/qwen3-next-80b-a3b-instruct:free", None),
    ("tencent/hy3:free", {"enabled": False}),                  # Hy3, non-think
]

# Where the report lands: JOBSCOUT_REPORT_DIR if set, else ./bakeoff_reports/
# (created on demand). Avoids hardcoding any one machine's temp/scratch path.
REPORT_DIR = Path(os.environ.get("JOBSCOUT_REPORT_DIR") or "bakeoff_reports")


class Result:
    __slots__ = (
        "fit", "reported_fit", "rationale", "flags", "breakdown",
        "in_tok", "out_tok", "cost", "secs", "error",
    )

    def __init__(self, fit=None, reported_fit=None, rationale="", flags=None,
                 breakdown=None, in_tok=0, out_tok=0, cost=0.0, secs=0.0, error=""):
        self.fit = fit
        # The model's self-reported fit_score, kept alongside the finalized fit so
        # Tier-1.1 can catch arithmetic drift (fit_score != 50 + Σ breakdown).
        self.reported_fit = reported_fit
        self.rationale = rationale
        self.flags = flags or []
        self.breakdown = breakdown or {}
        self.in_tok = in_tok
        self.out_tok = out_tok
        self.cost = cost
        self.secs = secs
        self.error = error


def _finalize_fit(data: dict, cfg: Config) -> int:
    """Mirror BulkScorer.score's fit computation, including the domain ceiling."""
    breakdown = data.get("breakdown", {})
    if breakdown:
        fit = max(0, min(100, 50 + sum(breakdown.values())))
        ceiling = cfg.scoring.specialist_domain_ceiling
        if ceiling < 100 and breakdown.get("domain", 0) <= -5:
            fit = min(fit, ceiling)
        return fit
    return int(data["fit_score"])


def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _ok_result(data: dict, cfg: Config, in_tok: int, out_tok: int, cost: float, secs: float) -> Result:
    """Build a successful Result from parsed JSON, capturing the raw self-reported
    fit_score, flags, and breakdown so the Tier-1 scorecard can audit them."""
    return Result(
        fit=_finalize_fit(data, cfg),
        reported_fit=_as_int(data.get("fit_score")),
        rationale=data.get("rationale", ""),
        flags=data.get("flags") or [],
        breakdown=data.get("breakdown") or {},
        in_tok=in_tok,
        out_tok=out_tok,
        cost=cost,
        secs=secs,
    )


def _cost(model: str, in_tok: int, out_tok: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    # OpenRouter billing for Anthropic models includes provider cache accounting.
    return (
        in_tok / 1e6 * pin
        + cache_write / 1e6 * pin * 1.25
        + cache_read / 1e6 * pin * 0.10
        + out_tok / 1e6 * pout
    )


OPENAI_EFFORT = "low"  # scoring is structured, not a reasoning olympiad — keep tokens down


def score_openai(client, model: str, system: str, listing: Listing, cfg: Config, effort: str = OPENAI_EFFORT) -> Result:
    # GPT-5.6 is a reasoning family; the Responses API is the modern surface.
    user = f"Score this listing:\n\n{_listing_text(listing)}"
    t0 = time.time()
    try:
        resp = client.responses.create(
            model=model,
            instructions=system,
            input=user,
            reasoning={"effort": effort},
        )
    except Exception as exc:
        return Result(error=f"{type(exc).__name__}: {exc}", secs=time.time() - t0)
    secs = time.time() - t0
    raw = getattr(resp, "output_text", "") or ""
    u = getattr(resp, "usage", None)
    in_tok = getattr(u, "input_tokens", 0) if u else 0
    out_tok = getattr(u, "output_tokens", 0) if u else 0
    # Credit OpenAI's automatic prompt caching: cached input bills at 0.5x (not
    # Anthropic's 0.1x). Without this the shared system prompt is over-billed —
    # the asymmetry that made the first run unfair to Terra.
    cached = 0
    details = getattr(u, "input_tokens_details", None) if u else None
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    pin, pout = PRICES.get(model, (0.0, 0.0))
    non_cached = max(0, in_tok - cached)
    cost = non_cached / 1e6 * pin + cached / 1e6 * pin * 0.5 + out_tok / 1e6 * pout
    try:
        data = _parse_score(raw)
        return _ok_result(data, cfg, in_tok, out_tok, cost, secs)
    except Exception as exc:
        return Result(rationale=f"[parse error: {exc}] {raw[:120]}",
                      in_tok=in_tok, out_tok=out_tok, cost=cost, secs=secs, error="parse")


def _with_retries(fn, label: str, max_retries: int = 5):
    """Retry on rate limits (429) with exponential backoff, honouring Retry-After.
    No fallback models — a persistent failure is raised for the caller to record."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            msg = str(exc).lower()
            is_rate = status == 429 or "429" in msg or ("rate" in msg and "limit" in msg)
            if not (is_rate and attempt < max_retries):
                raise
            retry_after = None
            resp = getattr(exc, "response", None)
            if resp is not None and hasattr(resp, "headers"):
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                try:
                    retry_after = float(ra) if ra else None
                except (TypeError, ValueError):
                    retry_after = None
            wait = retry_after if retry_after is not None else delay
            print(f" [rate-limited on {label}: waiting {wait:.0f}s]", end="", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 30)


def _or_cost(usage) -> float:
    """OpenRouter returns the real charged cost on usage.cost when we ask for it."""
    if usage is None:
        return 0.0
    c = getattr(usage, "cost", None)
    if c is None and getattr(usage, "model_extra", None):
        c = usage.model_extra.get("cost")
    try:
        return float(c or 0.0)
    except (TypeError, ValueError):
        return 0.0


def score_openrouter(client, model: str, reasoning, system: str, listing: Listing, cfg: Config) -> Result:
    # OpenRouter is OpenAI-compatible via chat.completions. usage.include returns the
    # real charged cost (free models report 0); reasoning controls thinking per model.
    user = f"Score this listing:\n\n{_listing_text(listing)}"
    extra_body: dict = {"usage": {"include": True}}
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    t0 = time.time()
    try:
        resp = _with_retries(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body=extra_body,
            ),
            label=model,
        )
    except Exception as exc:
        return Result(error=f"{type(exc).__name__}: {exc}", secs=time.time() - t0)
    secs = time.time() - t0
    raw = (resp.choices[0].message.content or "") if resp.choices else ""
    u = getattr(resp, "usage", None)
    in_tok = getattr(u, "prompt_tokens", 0) if u else 0
    out_tok = getattr(u, "completion_tokens", 0) if u else 0
    cost = _or_cost(u)
    try:
        data = _parse_score(raw)
        return _ok_result(data, cfg, in_tok, out_tok, cost, secs)
    except Exception as exc:
        return Result(rationale=f"[parse error: {exc}] {raw[:120]}",
                      in_tok=in_tok, out_tok=out_tok, cost=cost, secs=secs, error="parse")


# ---------------------------------------------------------------------------
# Tier-1.1 — rubric-internal consistency checks. Pure post-processors on the JSON
# a model already returned: no gold label needed. Each violation is a place where
# the model contradicts the rubric it was handed, which is a provable error.
# ---------------------------------------------------------------------------
def _dimension_ranges(cfg: Config) -> dict[str, tuple[int, int]]:
    """Allowed (min, max) for each breakdown term, mirroring the ranges the
    scoring prompt hands the model (see jobscout/scoring.py)."""
    w = cfg.scoring.weights
    s = cfg.scoring
    ranges = {
        "title": (0, w.title),
        "scope": (0, w.scope),
        "company": (0, w.company),
        "stack": (0, w.stack),
        "domain": (-w.domain, 0),
        "dealbreakers": (-w.dealbreakers, 0),
        # salary/office are graded on a -20..0 scale in the prompt (the weight is
        # only the mid "not stated" / "at your limit" penalty); 0 when unset.
        "salary": (-20, 0) if s.salary_floor is not None else (0, 0),
        "office": (-20, 0) if s.max_office_days is not None else (0, 0),
    }
    return ranges


def _consistency_violations(r: Result, cfg: Config) -> list[str]:
    """Return the rubric self-contradictions in one output (empty = clean)."""
    if r.fit is None or not r.breakdown:
        return []
    bd = r.breakdown
    v: list[str] = []

    # 1. Arithmetic: fit_score must equal 50 + Σ(breakdown), capped 0-100. When a
    #    specialist-domain ceiling is active it may legitimately pull the score
    #    down, so allow that value too.
    if r.reported_fit is not None:
        try:
            expected = max(0, min(100, 50 + int(sum(bd.values()))))
        except (TypeError, ValueError):
            expected = None
        if expected is not None:
            allowed = {expected}
            ceiling = cfg.scoring.specialist_domain_ceiling
            if ceiling < 100 and bd.get("domain", 0) <= -5:
                allowed.add(min(expected, ceiling))
            if r.reported_fit not in allowed:
                v.append(f"arithmetic(fit={r.reported_fit}!={expected})")

    # 2. Each term inside its allowed range.
    ranges = _dimension_ranges(cfg)
    for key, val in bd.items():
        lo, hi = ranges.get(key, (-100, 100))
        if not isinstance(val, (int, float)) or not (lo <= val <= hi):
            v.append(f"range:{key}={val}!in[{lo},{hi}]")

    # 3. Flag <-> score coherence.
    flags = [str(f).lower() for f in r.flags]
    if any("dealbreaker" in f for f in flags) and bd.get("dealbreakers", 0) >= 0:
        v.append("flag:dealbreaker-not-penalised")
    if any("skills-mismatch-required" in f for f in flags) and bd.get("stack", 0) > cfg.scoring.weights.stack / 2:
        v.append("flag:skills-mismatch-but-high-stack")

    return v


def _ok_runs(runs: list[Result]) -> list[Result]:
    return [r for r in runs if r.fit is not None]


def sample_listing_ids(store: Store, n: int) -> list[int]:
    """Stratified sample across high/mid/low fit buckets so we see behaviour on
    good, borderline, and poor fits rather than only one end."""
    rows = store.conn.execute(
        "SELECT listing_id, MAX(fit_score) f FROM scores GROUP BY listing_id"
    ).fetchall()
    buckets = {"high": [], "mid": [], "low": []}
    for lid, f in rows:
        b = "high" if f >= 70 else "mid" if f >= 50 else "low"
        buckets[b].append((lid, f))
    per = max(1, n // 3)
    picked: list[int] = []
    for b in ("high", "mid", "low"):
        col = sorted(buckets[b], key=lambda r: r[1])
        step = max(1, len(col) // per) if col else 1
        picked += [lid for lid, _ in col[::step][:per]]
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="sample size (default 15, spread across buckets)")
    ap.add_argument("--ids", type=int, nargs="*", help="explicit listing ids instead of a sample")
    ap.add_argument("--openrouter-models", nargs="*", help="override OpenRouter candidate models")
    ap.add_argument("--openai", action="store_true", help="also run the OpenAI arm (needs openai + OPENAI_API_KEY)")
    ap.add_argument("--openai-models", nargs="*", default=OPENAI_MODELS_DEFAULT)
    ap.add_argument("--repeats", type=int, default=1,
                    help="score each listing N times per model for test-retest reliability (Tier-1.3)")
    ap.add_argument("--smoke", action="store_true", help="score 1 listing with the baseline model only (key/balance check)")
    args = ap.parse_args()

    cfg = load_config()
    profile = build_profile(cfg)
    system = _build_system(cfg, profile)

    baseline = cfg.ai.bulk_model
    # Explicit --openrouter-models wins. Otherwise run the default OpenRouter set
    # only when no other arm was requested — so `--openai` alone runs just those models.
    if args.openrouter_models is not None:
        openrouter_models = list(dict.fromkeys(args.openrouter_models))
    elif args.openai:
        openrouter_models = []
    else:
        openrouter_models = list(dict.fromkeys(
            [m for m, _ in OPENROUTER_MODELS]
        ))

    orclient = None
    or_reasoning: dict[str, dict | None] = dict(OPENROUTER_MODELS)
    if openrouter_models or args.smoke:
        try:
            orclient = openrouter_client(cfg)
        except Exception as exc:
            print(f"[openrouter arm skipped: {exc}]")
            openrouter_models = []
            if args.smoke:
                return

    with Store(cfg.store.db_path) as store:
        total_listings = store.conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]

        if args.smoke:
            lid = sample_listing_ids(store, 1)[0]
            listing = store.get_listing(lid)
            print(f"Smoke test: scoring listing {lid} ({listing.title} @ {listing.company}) with {baseline}…")
            r = score_openrouter(orclient, baseline, or_reasoning.get(baseline), system, listing, cfg)
            if r.error and r.error != "parse":
                print(f"  ✗ API call failed: {r.error}")
            else:
                print(f"  ✓ fit={r.fit}  in={r.in_tok} out={r.out_tok}  ${r.cost:.4f}  {r.secs:.1f}s")
                print(f"    {r.rationale[:160]}")
            return

        ids = args.ids or sample_listing_ids(store, args.n)
        listings = [store.get_listing(i) for i in ids]
        listings = [job for job in listings if job]
        baselines = {job.id: (store.get_best_score(job.id).fit_score if store.get_best_score(job.id) else None) for job in listings}

        oclient = None
        openai_models: list[str] = []
        if args.openai:
            try:
                import openai
                oclient = openai.OpenAI()
                openai_models = args.openai_models
            except Exception as exc:
                print(f"[openai arm skipped: {exc}]")

        columns = (
            [("openrouter", m) for m in openrouter_models]
            + [("openai", m) for m in openai_models]
        )
        results: dict[str, dict[int, list[Result]]] = {m: {} for _, m in columns}

        repeats = max(1, args.repeats)
        print(f"Scoring {len(listings)} listings × {len(columns)} models"
              + (f" × {repeats} repeats" if repeats > 1 else "")
              + f" (baseline stored scores from {baseline})\n")
        for kind, spec in columns:
            print(f"  {spec} …", end="", flush=True)
            for job in listings:
                runs: list[Result] = []
                for _ in range(repeats):
                    if kind == "openrouter":
                        runs.append(score_openrouter(orclient, spec, or_reasoning.get(spec), system, job, cfg))
                    else:  # openai
                        # OpenAI specs may carry an effort suffix, e.g. "gpt-5.6-luna:high".
                        model_id, _, effort = spec.partition(":")
                        runs.append(score_openai(oclient, model_id, system, job, cfg, effort or OPENAI_EFFORT))
                results[spec][job.id] = runs
            errs = sum(1 for runs in results[spec].values() for r in runs if r.error and r.error != "parse")
            print(" done" + (f" ({errs} errors)" if errs else ""))

        _report(cfg, listings, baselines, columns, results, baseline, total_listings, repeats)


def _f1(x) -> str:
    return f"{x:.1f}" if x is not None else "—"


def _f2(x) -> str:
    return f"{x:.2f}" if x is not None else "—"


def _f4(x) -> str:
    return f"{x:.4f}" if x is not None else "—"


def _pct(x) -> str:
    return f"{x:.0f}%" if x is not None else "—"


def _build_scorecard(cfg, listings, baselines, models, results, total_listings):
    """Aggregate the per-run Results into one Tier-1 scorecard row per model."""
    card: dict[str, dict] = {}
    for m in models:
        all_runs = [r for job in listings for r in results[m].get(job.id, [])]
        ok = [r for r in all_runs if r.fit is not None]
        out = len(all_runs)

        # Per-listing representative fit (mean over successful repeats) + reliability.
        listing_fits: dict[int, float] = {}
        sigmas: list[float] = []
        for job in listings:
            okr = _ok_runs(results[m].get(job.id, []))
            if okr:
                fits = [r.fit for r in okr]
                listing_fits[job.id] = sum(fits) / len(fits)
                if len(fits) >= 2:
                    sigmas.append(statistics.pstdev(fits))

        mean_fit = sum(listing_fits.values()) / len(listing_fits) if listing_fits else None
        deltas = [abs(listing_fits[jid] - baselines[jid]) for jid in listing_fits
                  if baselines[jid] is not None]
        mean_delta = sum(deltas) / len(deltas) if deltas else None

        # Consistency (Tier-1.1) over successful outputs only.
        viols = [_consistency_violations(r, cfg) for r in ok]
        total_viol = sum(len(v) for v in viols)
        clean = sum(1 for v in viols if not v)
        viol_per_out = total_viol / len(ok) if ok else None
        clean_pct = 100 * clean / len(ok) if ok else None
        mean_sigma = sum(sigmas) / len(sigmas) if sigmas else None

        # Cost of ONE scoring pass = mean per-listing cost over repeats, summed.
        cost_sample = sum(
            (sum(r.cost for r in runs) / len(runs))
            for job in listings if (runs := results[m].get(job.id))
        )
        cost_each = cost_sample / len(listings) if listings else 0.0

        card[m] = dict(
            out=out, ok=len(ok), parse_err=sum(1 for r in all_runs if r.error == "parse"),
            mean_fit=mean_fit, mean_delta=mean_delta, viol_per_out=viol_per_out,
            clean_pct=clean_pct, total_viol=total_viol, sigma=mean_sigma,
            cost_each=cost_each, proj=cost_each * total_listings,
        )
    return card


def _report(cfg, listings, baselines, columns, results, baseline_model, total_listings, repeats=1):
    models = [m for _, m in columns]
    card = _build_scorecard(cfg, listings, baselines, models, results, total_listings)

    # --- console scorecard ---
    print("\n" + "=" * 94)
    print(f"TIER-1 AUTOMATED SCORECARD  ({len(listings)} listings × {repeats} repeat(s); "
          f"lower Δbase / viol / σ is better)")
    print(f"{'model':<26}{'mean':>6}{'Δbase':>7}{'viol/o':>8}{'clean':>7}"
          f"{'σ':>6}{'parse':>7}{'$/list':>10}{'$ all':>9}")
    print("-" * 94)
    for m in models:
        s = card[m]
        if s["ok"] == 0:
            print(f"{m:<26}{'— all outputs failed —':>68}")
            continue
        sigma = _f2(s["sigma"]) if repeats > 1 else "—"
        parse = _pct(100 * s["parse_err"] / s["out"] if s["out"] else None)
        print(f"{m:<26}{_f1(s['mean_fit']):>6}{_f1(s['mean_delta']):>7}"
              f"{_f2(s['viol_per_out']):>8}{_pct(s['clean_pct']):>7}{sigma:>6}"
              f"{parse:>7}{_f4(s['cost_each']):>10}{_f2(s['proj']):>9}")
    print("=" * 94)
    print("viol/o = mean rubric self-contradictions per output (arithmetic / range / flag). "
          "clean = % with none.")
    print("σ = mean per-listing score std across repeats (test-retest reliability; lower = steadier).")
    print(f"$ all = projected cost to score all {total_listings} listings once on that model.\n")

    # --- detailed markdown ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = REPORT_DIR / "bakeoff_report.md"
    L: list[str] = ["# Scoring model bake-off\n"]
    L.append(f"Baseline stored scores produced by `{baseline_model}`. "
             f"{len(listings)} listings × {repeats} repeat(s).\n")

    L.append("## Tier-1 automated scorecard\n")
    L.append("| model | mean fit | Δ base | viol/out | clean % | σ (retest) | parse err | $/listing | $ all corpus |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for m in models:
        s = card[m]
        sigma = _f2(s["sigma"]) if repeats > 1 else "—"
        L.append(
            f"| {m} | {_f1(s['mean_fit'])} | {_f1(s['mean_delta'])} | {_f2(s['viol_per_out'])} | "
            f"{_pct(s['clean_pct'])} | {sigma} | {s['parse_err']}/{s['out']} | "
            f"{_f4(s['cost_each'])} | {_f2(s['proj'])} |"
        )
    L.append("")
    L.append("- **Δ base**: mean absolute gap vs your stored score (agreement, not correctness).")
    L.append("- **viol/out**: mean rubric self-contradictions per output — arithmetic (fit≠50+Σ), "
             "out-of-range term, or flag↔score incoherence. A model contradicting its own rubric is "
             "making provable errors, no gold label required.")
    L.append("- **σ (retest)**: mean per-listing score std across repeats; lower = steadier shortlist.")
    L.append("")

    # Consistency-violation detail (Tier-1.1).
    L.append("## Consistency violations (Tier-1.1)\n")
    any_viol = False
    for m in models:
        rows = []
        for job in listings:
            for i, r in enumerate(_ok_runs(results[m].get(job.id, []))):
                v = _consistency_violations(r, cfg)
                if v:
                    tag = f"[{job.id}]" + (f" run{i + 1}" if repeats > 1 else "")
                    rows.append(f"  - {tag}: {', '.join(v)}")
        if rows:
            any_viol = True
            L.append(f"**{m}** — {len(rows)} flagged output(s):")
            L.extend(rows)
            L.append("")
    if not any_viol:
        L.append("None — every model's outputs were internally consistent with the rubric.\n")

    # Test–retest reliability detail (Tier-1.3), only when repeated.
    if repeats > 1:
        L.append("## Test–retest reliability (Tier-1.3)\n")
        L.append("Per-listing fit across repeats (σ = std). Wide spread = same input, unstable score.\n")
        for m in models:
            L.append(f"### {m}")
            L.append("| listing | fits | σ |")
            L.append("|---|---|--:|")
            for job in listings:
                fits = [r.fit for r in _ok_runs(results[m].get(job.id, []))]
                sig = statistics.pstdev(fits) if len(fits) >= 2 else None
                fits_s = ", ".join(str(f) for f in fits) if fits else "—"
                L.append(f"| [{job.id}] | {fits_s} | {_f2(sig)} |")
            L.append("")

    # Side-by-side rationales for eyeballing.
    L.append("## Side-by-side (fit · flags · violations · rationale)\n")
    for job in listings:
        L.append(f"### [{job.id}] {job.title} @ {job.company}")
        L.append(f"<{job.url}>  ·  baseline stored: **{baselines[job.id]}**\n")
        L.append("| model | fit | flags | violations | rationale |")
        L.append("|---|--:|---|---|---|")
        for m in models:
            runs = results[m].get(job.id, [])
            okr = _ok_runs(runs)
            if okr:
                fits = [r.fit for r in okr]
                fit_disp = f"{sum(fits) / len(fits):.0f} ({min(fits)}–{max(fits)})" if len(fits) > 1 else str(fits[0])
                rep = okr[0]
                v = _consistency_violations(rep, cfg)
                flags = ", ".join(str(f) for f in rep.flags)[:60]
                viol_s = "; ".join(v)[:80] if v else "✓"
                rat = (rep.rationale or "").replace("\n", " ")[:200]
            else:
                fit_disp, flags, viol_s = "ERR", "", ""
                rat = next((r.rationale or r.error for r in runs), "").replace("\n", " ")[:200]
            L.append(f"| {m} | {fit_disp} | {flags} | {viol_s} | {rat} |")
        L.append("")

    md.write_text("\n".join(L))
    print(f"Detailed report (scorecard + consistency + reliability + rationales) written to:\n  {md}")


if __name__ == "__main__":
    main()
