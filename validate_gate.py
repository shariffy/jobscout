#!/usr/bin/env python3
"""Regression gate: every role you've applied to must still clear the shortlist.

Your applications are ground-truth positive labels — you looked at these roles and
decided they were worth applying to. After any change to gates/priority/profile this
re-scores them and fails loudly if the change would have hidden any from you.

APPLIED and PREPPING roles (minus the disowned set below) must all get
decision == "apply". Negative labels (known dealbreakers) must get
decision == "no" — and fail the expected gate when one is named.

    .venv/bin/python validate_gate.py
    .venv/bin/python validate_gate.py --repeats 3
    .venv/bin/python validate_gate.py --negative 123 --negative 456:office
    .venv/bin/python validate_gate.py --model deepseek/deepseek-v4-pro
"""
from __future__ import annotations

import argparse
from collections import Counter

from jobscout.assess import GatedScorer
from jobscout.config import load_config
from jobscout.store import Store

# Roles you applied to / were prepping but have since disowned as labels —
# they no longer represent "I would apply to this".
DISOWNED: set[int] = {466}

# Known-dealbreaker roles: listing_id -> expected failing gate name ("" = any gate).
NEGATIVE_LABELS: dict[int, str] = {
    23:  "no_title_specialization",
    72:  "no_data_focus",
    73:  "no_fde",
    76:  "no_data_focus",
    129: "no_data_focus",
    173: "no_fde",
    178: "no_blocked_stack",
    219: "no_title_specialization",
    234: "no_blocked_stack",
    260: "no_blocked_stack",
    279: "no_data_focus",
    304: "no_title_specialization",
    314: "no_title_specialization",
    434: "no_ml_platform",
    892: "no_blocked_stack",
    898: "industry",
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


def run_gated(cfg, args) -> None:
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
          f"{'gate':>6}  title / failed gates")
    print("-" * 95)

    failures: list[int] = []
    for lid, status, title, listing, want in jobs:
        decisions: list[str] = []
        last = None
        try:
            for _ in range(repeats):
                last = scorer.score(listing)
                decisions.append(last.decision)
        except Exception as exc:
            print(f"{lid:>5} {status:<9} {want:>6} {'ERR':>6} {'—':>5} {'—':>5} "
                  f"{'FAIL':>6}  {title[:35]}  [{exc}]")
            failures.append(lid)
            continue

        got = Counter(decisions).most_common(1)[0][0]
        failed_gates = [f.removeprefix("gate-fail-") for f in last.flags
                        if f.startswith("gate-fail-")]
        ok = got == want
        expected_gate = negatives.get(lid, "")
        if ok and want == "no" and expected_gate and expected_gate not in failed_gates:
            ok = False
        gate = "ok" if ok else "FAIL"
        if ok and len(set(decisions)) > 1:
            gate = "ok!"
        if not ok:
            failures.append(lid)

        prio = last.priority if last.priority is not None else "—"
        detail = title[:35]
        if failed_gates:
            detail += f"  [{', '.join(failed_gates)}]"
        print(f"{lid:>5} {status:<9} {want:>6} {got:>6} {last.tier_label or '—':>5} "
              f"{str(prio):>5} {gate:>6}  {detail}")

    print("-" * 95)
    print("gate: ok = decision matches label · ok! = majority right but runs disagreed · "
          "FAIL = wrong decision\n")
    if failures:
        print(f"❌ GATE FAILED — {len(failures)} role(s) mislabelled: {failures}")
        raise SystemExit(1)
    print("✅ GATE PASSED — every labelled role gets the right decision.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1,
                    help="score each role N times; gate on the majority decision")
    ap.add_argument("--model", default=None, help="model slug (default: cfg.ai.bulk_model)")
    ap.add_argument("--negative", action="append", metavar="ID[:GATE]",
                    help="listing id that must get decision=no (optionally via GATE)")
    args = ap.parse_args()
    run_gated(load_config(), args)


if __name__ == "__main__":
    main()
