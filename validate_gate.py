#!/usr/bin/env python3
"""Regression gate: every role you've applied to must still clear the shortlist.

Your applications are ground-truth positive labels — you looked at these roles and
decided they were worth applying to. After any change to scoring/gates/profile this
re-scores them and fails loudly if the change would have hidden any from you.

Two modes, selected by --scorer (default: ai.scorer from config.toml):

  additive  Legacy 0-100 fit. APPLIED roles must score >= fit_threshold
            (prepping shown with --include-prepping, not gated).
  gated     APPLIED and PREPPING roles (minus the disowned set below) must all get
            decision == "apply". Negative labels (known dealbreakers) must get
            decision == "no" — and fail the expected gate when one is named.

    .venv/bin/python validate_gate.py                          # mode from config
    .venv/bin/python validate_gate.py --scorer gated --repeats 3
    .venv/bin/python validate_gate.py --scorer gated --negative 123 --negative 456:office
    .venv/bin/python validate_gate.py --model deepseek/deepseek-v4-pro

Needs OPEN_ROUTER_API_KEY in the environment (same as bakeoff.py).
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from jobscout.config import load_config
from jobscout.store import Store

# Roles you applied to / were prepping but have since disowned as labels —
# they no longer represent "I would apply to this".
DISOWNED: set[int] = {466}

# Known-dealbreaker roles: listing_id -> expected failing gate name ("" = any gate).
# Kept in-file so the negative set is versioned with the gate rules it tests.
#
# Sourced from roles you marked not_interested in Notion — each a genuine hard
# dealbreaker (not a soft "didn't grab me" pass, which the gate is meant to APPLY and
# rank low). Where a role trips several gates, the expected name is its primary reason.
NEGATIVE_LABELS: dict[int, str] = {
    23:  "no_title_specialization",  # Wildcard — applied ML engineer (ML in title)
    72:  "no_data_focus",            # Head of Analytics Engineering
    73:  "no_fde",                   # Head of Client Engineering (implementation)
    76:  "no_data_focus",            # Head of Analytics Engineering
    129: "no_data_focus",            # EM - Data Platform (also no_blocked_stack)
    173: "no_fde",                   # Senior Manager, Solutions Engineering
    178: "no_blocked_stack",         # Senior EM - Compliance
    219: "no_title_specialization",  # AI Technical Lead - C++
    234: "no_blocked_stack",         # Principal Backend Engineer
    260: "no_blocked_stack",         # Senior Staff SWE - Risk
    279: "no_data_focus",            # Principal/Staff Engineer (Data)
    304: "no_title_specialization",  # AI Engineering Manager (AI in title)
    314: "no_title_specialization",  # ML Technical Lead (ML in title)
    434: "no_ml_platform",           # EM (AI Enablement)
    892: "no_blocked_stack",         # FYST — Go as primary stack
    898: "industry",                 # Atominvest — PE/investment software
}


def _ground_truth(store: Store, statuses: list[str]) -> list[tuple[int, str, str]]:
    qmarks = ",".join("?" * len(statuses))
    rows = store.conn.execute(
        f"""SELECT a.listing_id, a.status, l.title
            FROM applications a JOIN listings l ON l.id = a.listing_id
            WHERE a.status IN ({qmarks})
            ORDER BY a.status, a.listing_id""",
        statuses,
    ).fetchall()
    return [(r["listing_id"], r["status"], r["title"]) for r in rows]


def run_additive(cfg, args) -> None:
    from bakeoff import OPENROUTER_MODELS, _consistency_violations, score_openrouter
    from jobscout.llm import openrouter_client
    from jobscout.profile import build_profile
    from jobscout.scoring import _build_system

    profile = build_profile(cfg)
    system = _build_system(cfg, profile)
    client = openrouter_client(cfg)
    model = args.model or cfg.ai.bulk_model
    # Match bakeoff's per-model reasoning config so the gate reflects real scoring.
    reasoning = dict(OPENROUTER_MODELS).get(model)
    threshold = cfg.ai.fit_threshold
    repeats = max(1, args.repeats)

    statuses = ["applied"] + (["prepping"] if args.include_prepping else [])
    with Store(cfg.store.db_path) as store:
        jobs = [
            (lid, status, title, store.get_listing(lid), store.get_best_score(lid))
            for lid, status, title in _ground_truth(store, statuses)
        ]

    print(f"Gate: APPLIED roles must score >= {threshold} on {model}"
          + (f"  (mean of {repeats} runs)" if repeats > 1 else "") + "\n")
    print(f"{'id':>5} {'status':<9} {'was':>4} {'now':>12} {'gate':>6}  title")
    print("-" * 78)

    failures: list[int] = []
    for lid, status, title, listing, best in jobs:
        fits: list[int] = []
        viol = 0
        for _ in range(repeats):
            r = score_openrouter(client, model, reasoning, system, listing, cfg)
            if r.fit is not None:
                fits.append(r.fit)
                viol += len(_consistency_violations(r, cfg))

        if not fits:
            now, gate = "ERR", "FAIL"
        else:
            mean = statistics.mean(fits)
            lo, hi = min(fits), max(fits)
            now = f"{mean:.0f}" + (f" ({lo}-{hi})" if repeats > 1 else "")
            # Gate on the mean; mark borderline when a run dipped under the line.
            passed = mean >= threshold
            gate = "ok" if passed else "FAIL"
            if passed and lo < threshold:
                gate = "ok!"  # mean clears it but at least one run dipped below
        if gate.startswith("FAIL") and status == "applied":
            failures.append(lid)
        was = best.fit_score if best else "—"
        vflag = f"  ⚠{viol} viol" if viol else ""
        print(f"{lid:>5} {status:<9} {str(was):>4} {now:>12} {gate:>6}  {title[:40]}{vflag}")

    print("-" * 78)
    print("gate: ok = mean >= threshold · ok! = passes but a run dipped below · "
          "FAIL = below (applied only)\n")
    if failures:
        print(f"❌ GATE FAILED — {len(failures)} applied role(s) below {threshold}: {failures}")
        raise SystemExit(1)
    print("✅ GATE PASSED — every applied role still clears the shortlist threshold.")


def run_gated(cfg, args) -> None:
    from jobscout.assess import GatedScorer

    if args.model:
        cfg.ai.bulk_model = args.model
    scorer = GatedScorer(cfg)
    repeats = max(1, args.repeats)

    negatives = dict(NEGATIVE_LABELS)
    for spec in args.negative or []:
        lid, _, gate = spec.partition(":")
        negatives[int(lid)] = gate

    with Store(cfg.store.db_path) as store:
        positives = [
            (lid, status, title)
            for lid, status, title in _ground_truth(store, ["applied", "prepping"])
            if lid not in DISOWNED
        ]
        jobs = [(lid, status, title, store.get_listing(lid), "apply")
                for lid, status, title in positives]
        for lid, gate_name in negatives.items():
            listing = store.get_listing(lid)
            if listing is None:
                print(f"[warn] negative label {lid} not in DB — skipped")
                continue
            jobs.append((lid, "negative", listing.title, listing, "no"))

    print(f"Gate: {len(positives)} ground-truth role(s) must get decision=apply, "
          f"{len(negatives)} negative(s) must get decision=no, on {cfg.ai.bulk_model}"
          + (f"  (majority of {repeats} runs)" if repeats > 1 else "") + "\n")
    print(f"{'id':>5} {'status':<9} {'want':>6} {'got':>6} {'tier':>5} {'prio':>5} "
          f"{'fit':>4} {'gate':>6}  title / failed gates")
    print("-" * 100)

    failures: list[int] = []
    for lid, status, title, listing, want in jobs:
        decisions: list[str] = []
        last = None
        try:
            for _ in range(repeats):
                last = scorer.score(listing)
                decisions.append(last.decision)
        except Exception as exc:
            print(f"{lid:>5} {status:<9} {want:>6} {'ERR':>6} {'—':>5} {'—':>5} {'—':>4} "
                  f"{'FAIL':>6}  {title[:35]}  [{exc}]")
            failures.append(lid)
            continue

        # Majority vote on the decision across repeats (self-consistency).
        got = Counter(decisions).most_common(1)[0][0]
        failed_gates = [f.removeprefix("gate-fail-") for f in last.flags
                        if f.startswith("gate-fail-")]
        ok = got == want
        expected_gate = negatives.get(lid, "")
        if ok and want == "no" and expected_gate and expected_gate not in failed_gates:
            ok = False  # rejected, but for the wrong reason
        gate = "ok" if ok else "FAIL"
        if ok and len(set(decisions)) > 1:
            gate = "ok!"  # majority is right but runs disagreed
        if not ok:
            failures.append(lid)

        prio = last.priority if last.priority is not None else "—"
        detail = title[:35]
        if failed_gates:
            detail += f"  [{', '.join(failed_gates)}]"
        print(f"{lid:>5} {status:<9} {want:>6} {got:>6} {last.tier_label or '—':>5} "
              f"{str(prio):>5} {last.fit_score:>4} {gate:>6}  {detail}")

    print("-" * 100)
    print("gate: ok = decision matches label · ok! = majority right but runs disagreed · "
          "FAIL = wrong decision\n")
    if failures:
        print(f"❌ GATE FAILED — {len(failures)} role(s) mislabelled: {failures}")
        raise SystemExit(1)
    print("✅ GATE PASSED — every labelled role gets the right decision.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["additive", "gated"], default=None,
                    help="which producer to validate (default: ai.scorer from config)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="score each role N times; additive gates on the mean, "
                         "gated on the majority")
    ap.add_argument("--include-prepping", action="store_true",
                    help="additive mode: also show the 'prepping' set (shown, not gated)")
    ap.add_argument("--model", default=None, help="OpenRouter slug (default: cfg.ai.bulk_model)")
    ap.add_argument("--negative", action="append", metavar="ID[:GATE]",
                    help="gated mode: listing id that must get decision=no (optionally via GATE)")
    args = ap.parse_args()

    cfg = load_config()
    scorer = args.scorer or cfg.ai.scorer
    if scorer == "gated":
        run_gated(cfg, args)
    else:
        run_additive(cfg, args)


if __name__ == "__main__":
    main()
