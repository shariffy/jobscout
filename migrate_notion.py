#!/usr/bin/env python3
"""One-shot Phase-3 migration: add Decision/Priority/Tier to the Notion board and
backfill them on every linked page from the local best score. Run once after the
gated scorer has produced scores you're happy with; safe to re-run — it's
idempotent (patches the same properties again).

    .venv/bin/python migrate_notion.py --dry-run   # preview, no writes
    .venv/bin/python migrate_notion.py              # patch schema + backfill pages

Needs notion.token + notion.database_id in config.toml (same as the CLI).
"""
from __future__ import annotations

import argparse

import httpx

from jobscout.config import load_config
from jobscout.notion_sync import NotionSync
from jobscout.store import Store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run", action="store_true",
        help="show what would be written, don't patch Notion",
    )
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.notion.token or not cfg.notion.database_id:
        raise SystemExit(
            "Notion not configured (notion.token / notion.database_id in config.toml)."
        )
    ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)

    if not args.dry_run:
        print("Patching board schema (Decision / Priority / Tier)…")
        ns.update_schema()

    with Store(cfg.store.db_path) as store:
        rows = store.conn.execute(
            """
            SELECT a.listing_id, a.notion_page_id, l.title, l.company
            FROM applications a JOIN listings l ON l.id = a.listing_id
            WHERE a.notion_page_id != '' AND a.status != 'not_interested'
            ORDER BY a.listing_id
            """
        ).fetchall()

        migrated = 0
        skipped = 0
        failed = 0
        for r in rows:
            score = store.get_best_score(r["listing_id"])
            if not score:
                print(f"  - skip {r['listing_id']} (no score): {r['title']}")
                skipped += 1
                continue
            label = f"{score.decision or '—'} {score.tier_label or ''}".strip()
            if args.dry_run:
                print(f"  {label:>10}  {r['title']} @ {r['company']}")
                continue
            try:
                ns._patch(f"/pages/{r['notion_page_id']}", {"properties": ns.score_props(score)})
                migrated += 1
                print(f"  ✓ {label:>10}  {r['title']} @ {r['company']}")
            except httpx.HTTPStatusError as exc:
                # jobscout prune archives low-fit pages without clearing notion_page_id
                # or updating local status, and Notion refuses to edit archived blocks —
                # that's an expected steady-state skip, not a migration failure.
                if exc.response.status_code == 400 and "archived" in exc.response.text.lower():
                    print(f"  - skip {r['title']} (archived in Notion)")
                    skipped += 1
                else:
                    print(f"  ✗ {r['title']}: {exc.response.text[:200]}")
                    failed += 1
            except Exception as exc:
                print(f"  ✗ {r['title']}: {exc}")
                failed += 1

    if args.dry_run:
        print(
            f"\nDry run — {len(rows) - skipped} page(s) would be migrated "
            f"({skipped} skipped, no score)."
        )
    else:
        print(
            f"\nDone. {migrated}/{len(rows)} pages migrated "
            f"({skipped} skipped/archived, {failed} failed)."
        )


if __name__ == "__main__":
    main()
