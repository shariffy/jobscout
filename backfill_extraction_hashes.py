#!/usr/bin/env python3
"""One-time backfill: stamp existing extraction-cache rows with the content hash of
their listing's CURRENT text.

The extraction cache gained a text_hash column so an enriched JD invalidates a stale
extraction (jobscout/store.py, jobscout/scoring.py::content_hash). Rows written before
that change have text_hash='' and would miss on every lookup, needlessly re-paying the
LLM on the next rescore. This assigns each such row the hash of its listing's current
text, preserving the cache for the common case (text unchanged since extraction).

Caveat: a listing whose description was enriched but never rescored BEFORE this change
gets stamped as fresh here, so its (stale) extraction is trusted until the text next
changes. If you suspect any such listing, `jobscout rescore --all --force` re-extracts.

Idempotent: only rows with an empty text_hash are touched.

    .venv/bin/python backfill_extraction_hashes.py [--db jobscout.db] [--dry-run]
"""
from __future__ import annotations

import argparse

from jobscout.scoring import content_hash
from jobscout.store import Store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="jobscout.db")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()

    with Store(args.db) as store:
        rows = store.conn.execute(
            "SELECT id, listing_id FROM extractions WHERE text_hash = ''"
        ).fetchall()
        print(f"{len(rows)} extraction row(s) with no text_hash")

        updated = missing = 0
        for row in rows:
            listing = store.get_listing(row["listing_id"])
            if listing is None:
                missing += 1
                continue
            th = content_hash(listing)
            if not args.dry_run:
                store.conn.execute(
                    "UPDATE extractions SET text_hash = ? WHERE id = ?", (th, row["id"])
                )
            updated += 1
        if not args.dry_run:
            store.conn.commit()

        verb = "would stamp" if args.dry_run else "stamped"
        print(f"{verb} {updated} row(s); {missing} skipped (listing gone)")


if __name__ == "__main__":
    main()
