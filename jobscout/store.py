from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator

from .models import Application, Listing, Score

_SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    hash        TEXT    NOT NULL UNIQUE,
    source_name TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    company     TEXT    NOT NULL,
    location    TEXT    NOT NULL DEFAULT '',
    url         TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    raw         TEXT    NOT NULL DEFAULT '{}',
    fetched_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id         INTEGER NOT NULL UNIQUE REFERENCES listings(id) ON DELETE CASCADE,
    fit_score          INTEGER NOT NULL,
    rationale          TEXT    NOT NULL DEFAULT '',
    flags              TEXT    NOT NULL DEFAULT '[]',
    breakdown          TEXT    NOT NULL DEFAULT '{}',
    model              TEXT    NOT NULL,
    tier               TEXT    NOT NULL,
    decision           TEXT    NOT NULL DEFAULT '',
    priority           INTEGER,
    tier_label         TEXT    NOT NULL DEFAULT '',
    gate_results       TEXT    NOT NULL DEFAULT '[]',
    assessment_version TEXT    NOT NULL DEFAULT '',
    scored_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS extractions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id        INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    model             TEXT    NOT NULL,
    prompt_hash       TEXT    NOT NULL,
    repeat_idx        INTEGER NOT NULL DEFAULT 0,
    extraction        TEXT    NOT NULL,
    extracted_at      TEXT    NOT NULL,
    -- content fingerprint of the scored text; a cache hit also requires this to
    -- match so an enriched JD is re-read instead of served stale.
    text_hash         TEXT    NOT NULL DEFAULT '',
    -- real provider usage for the call that produced this row (null on cache reuse
    -- or when the route has no cost field, e.g. a direct provider API).
    cost              REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    UNIQUE(listing_id, model, prompt_hash, repeat_idx)
);

CREATE TABLE IF NOT EXISTS applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id      INTEGER NOT NULL UNIQUE REFERENCES listings(id),
    status          TEXT    NOT NULL DEFAULT 'shortlisted',
    applied_at      TEXT,
    chase_at        TEXT,
    notes           TEXT    NOT NULL DEFAULT '',
    contacts        TEXT    NOT NULL DEFAULT '',
    notion_page_id  TEXT    NOT NULL DEFAULT '',
    prep_content    TEXT    NOT NULL DEFAULT '',
    updated_at      TEXT    NOT NULL
);
"""


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Store:
    def __init__(self, db_path: str | Path = "jobscout.db") -> None:
        self._path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        # Guards the extraction cache, the one table touched concurrently: the
        # self-consistency vote extracts its mandatory repeats on worker threads
        # (see GatedScorer.score). Everything else runs single-threaded.
        self._extraction_lock = threading.Lock()

    def connect(self) -> None:
        # check_same_thread=False lets the vote's worker threads reach the cache;
        # concurrent access is confined to the two extraction methods and
        # serialised by _extraction_lock.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add assessment columns to a pre-gated scores table, then backfill.

        decision is derived from the fit >= 70 shortlist convention for any row
        that lacks one (legacy additive rows included), so `decision = 'apply'`
        and `fit_score >= 70` stay equivalent while both scorers coexist.
        Priority/tier are left empty until a gated rescore.
        """
        existing = {r["name"] for r in self.conn.execute("PRAGMA table_info(scores)")}
        for column, ddl in [
            ("decision", "decision TEXT NOT NULL DEFAULT ''"),
            ("priority", "priority INTEGER"),
            ("tier_label", "tier_label TEXT NOT NULL DEFAULT ''"),
            ("gate_results", "gate_results TEXT NOT NULL DEFAULT '[]'"),
            ("assessment_version", "assessment_version TEXT NOT NULL DEFAULT ''"),
        ]:
            if column not in existing:
                self.conn.execute(f"ALTER TABLE scores ADD COLUMN {ddl}")
        self.conn.execute(
            "UPDATE scores SET decision = CASE WHEN fit_score >= 70 THEN 'apply' ELSE 'no' END "
            "WHERE decision = ''"
        )
        if self._scores_needs_rebuild():
            self._rebuild_scores_table()

        # Extraction cache: content fingerprint + real usage. Existing rows keep
        # text_hash='' and are backfilled from current listing text by
        # backfill_extraction_hashes.py so they aren't needlessly re-extracted.
        ext_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(extractions)")}
        for column, ddl in [
            ("text_hash", "text_hash TEXT NOT NULL DEFAULT ''"),
            ("cost", "cost REAL"),
            ("prompt_tokens", "prompt_tokens INTEGER"),
            ("completion_tokens", "completion_tokens INTEGER"),
        ]:
            if column not in ext_cols:
                self.conn.execute(f"ALTER TABLE extractions ADD COLUMN {ddl}")

    def _scores_needs_rebuild(self) -> bool:
        """True if scores.listing_id lacks ON DELETE CASCADE or a UNIQUE constraint.

        Both are added by the same table rebuild (SQLite can't ALTER a foreign key
        or add a UNIQUE constraint in place), so one check covers both.
        """
        fks = self.conn.execute("PRAGMA foreign_key_list(scores)").fetchall()
        has_cascade = any(
            fk["table"] == "listings" and (fk["on_delete"] or "").upper() == "CASCADE"
            for fk in fks
        )
        has_unique_listing_id = False
        for idx in self.conn.execute("PRAGMA index_list(scores)").fetchall():
            if not idx["unique"]:
                continue
            cols = self.conn.execute(f"PRAGMA index_info({idx['name']})").fetchall()
            if [c["name"] for c in cols] == ["listing_id"]:
                has_unique_listing_id = True
                break
        return not (has_cascade and has_unique_listing_id)

    def _rebuild_scores_table(self) -> None:
        """Rebuild scores with ON DELETE CASCADE + UNIQUE(listing_id).

        SQLite can't ALTER a foreign key clause or add a constraint in place, so this
        copies rows into a freshly-defined table. Dedupes first (keep the newest row
        per listing_id, ties broken by id) since the UNIQUE constraint requires it.
        foreign_keys must be off for the RENAME/DROP sequence (SQLite cannot toggle it
        inside an open transaction, hence the commit before and after).
        """
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self.conn.execute("BEGIN")
            self.conn.execute(
                """
                DELETE FROM scores WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY listing_id ORDER BY scored_at DESC, id DESC
                        ) AS rn
                        FROM scores
                    ) WHERE rn = 1
                )
                """
            )
            self.conn.execute("ALTER TABLE scores RENAME TO scores_old")
            self.conn.execute(
                """
                CREATE TABLE scores (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    listing_id         INTEGER NOT NULL UNIQUE REFERENCES listings(id) ON DELETE CASCADE,
                    fit_score          INTEGER NOT NULL,
                    rationale          TEXT    NOT NULL DEFAULT '',
                    flags              TEXT    NOT NULL DEFAULT '[]',
                    breakdown          TEXT    NOT NULL DEFAULT '{}',
                    model              TEXT    NOT NULL,
                    tier               TEXT    NOT NULL,
                    decision           TEXT    NOT NULL DEFAULT '',
                    priority           INTEGER,
                    tier_label         TEXT    NOT NULL DEFAULT '',
                    gate_results       TEXT    NOT NULL DEFAULT '[]',
                    assessment_version TEXT    NOT NULL DEFAULT '',
                    scored_at          TEXT    NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                INSERT INTO scores (id, listing_id, fit_score, rationale, flags, breakdown,
                                     model, tier, decision, priority, tier_label, gate_results,
                                     assessment_version, scored_at)
                SELECT id, listing_id, fit_score, rationale, flags, breakdown,
                       model, tier, decision, priority, tier_label, gate_results,
                       assessment_version, scored_at
                FROM scores_old
                """
            )
            self.conn.execute("DROP TABLE scores_old")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.execute("PRAGMA foreign_keys=ON")
        # One-time reclaim: this rebuild runs once per DB (the constraint check
        # above short-circuits it thereafter), so VACUUM here is cheap over the
        # DB's lifetime but returns the space freed by dropping scores_old.
        self.conn.execute("VACUUM")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Store:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store is not connected — use as context manager or call connect()")
        return self._conn

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # --- listings ---

    def upsert_listing(self, listing: Listing) -> tuple[Listing, bool]:
        """Insert listing; return (listing_with_id, is_new). Dedups on hash.

        A reappearing hash whose new description is richer than what's stored
        (e.g. a source that fetches full JDs where a prior scan only got a card
        snippet) refreshes the stored description instead of discarding it.
        """
        row = self.conn.execute(
            "SELECT id, description FROM listings WHERE hash = ?", (listing.hash,)
        ).fetchone()

        if row:
            if listing.description and len(listing.description) > len(row["description"] or ""):
                self.update_listing_description(row["id"], listing.description)
            listing = listing.model_copy(update={"id": row["id"]})
            return listing, False

        cur = self.conn.execute(
            """
            INSERT INTO listings (hash, source_name, title, company, location, url, description, raw, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                listing.hash,
                listing.source_name,
                listing.title,
                listing.company,
                listing.location,
                listing.url,
                listing.description,
                json.dumps(listing.raw),
                _dt(listing.fetched_at),
            ),
        )
        self.conn.commit()
        listing = listing.model_copy(update={"id": cur.lastrowid})
        return listing, True

    def get_listing(self, listing_id: int) -> Listing | None:
        row = self.conn.execute("SELECT * FROM listings WHERE id = ?", (listing_id,)).fetchone()
        return _row_to_listing(row) if row else None

    def update_listing_description(self, listing_id: int, description: str) -> None:
        self.conn.execute(
            "UPDATE listings SET description = ? WHERE id = ?", (description, listing_id)
        )
        self.conn.commit()

    def list_listings(self, min_fit: int | None = None, limit: int = 100) -> list[dict]:
        """Return listings joined with their score, optionally filtered by fit."""
        sql = """
            SELECT l.*, s.fit_score, s.rationale, s.flags, s.tier,
                   s.decision, s.priority, s.tier_label
            FROM listings l
            LEFT JOIN scores s ON s.listing_id = l.id
        """
        params: list = []
        if min_fit is not None:
            sql += " WHERE s.fit_score >= ?"
            params.append(min_fit)
        sql += " ORDER BY s.fit_score DESC NULLS LAST, l.fetched_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def unscored_listings(self) -> list[Listing]:
        rows = self.conn.execute(
            """
            SELECT l.* FROM listings l
            WHERE NOT EXISTS (SELECT 1 FROM scores s WHERE s.listing_id = l.id)
            ORDER BY l.fetched_at DESC
            """
        ).fetchall()
        return [_row_to_listing(r) for r in rows]

    # --- scores ---

    def insert_score(self, score: Score) -> Score:
        """Upsert the score for score.listing_id — one row per listing (UNIQUE
        constraint), so a rescore replaces the prior score instead of appending."""
        # Legacy additive scores carry no decision; store the fit-derived one so
        # decision-based queries see every row.
        decision = score.decision or ("apply" if score.fit_score >= 70 else "no")
        self.conn.execute(
            """
            INSERT INTO scores (listing_id, fit_score, rationale, flags, breakdown, model, tier,
                                decision, priority, tier_label, gate_results, assessment_version,
                                scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(listing_id) DO UPDATE SET
                fit_score=excluded.fit_score,
                rationale=excluded.rationale,
                flags=excluded.flags,
                breakdown=excluded.breakdown,
                model=excluded.model,
                tier=excluded.tier,
                decision=excluded.decision,
                priority=excluded.priority,
                tier_label=excluded.tier_label,
                gate_results=excluded.gate_results,
                assessment_version=excluded.assessment_version,
                scored_at=excluded.scored_at
            """,
            (
                score.listing_id,
                score.fit_score,
                score.rationale,
                json.dumps(score.flags),
                json.dumps(score.breakdown),
                score.model,
                score.tier,
                decision,
                score.priority,
                score.tier_label,
                json.dumps(score.gate_results),
                score.assessment_version,
                _dt(score.scored_at),
            ),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM scores WHERE listing_id = ?", (score.listing_id,)
        ).fetchone()
        return score.model_copy(update={"id": row["id"], "decision": decision})

    def get_best_score(self, listing_id: int) -> Score | None:
        row = self.conn.execute(
            "SELECT * FROM scores WHERE listing_id = ?", (listing_id,)
        ).fetchone()
        return _row_to_score(row) if row else None

    # --- extractions (Stage-1 cache) ---

    def save_extraction(
        self, listing_id: int, model: str, prompt_hash: str, repeat_idx: int,
        extraction_json: str, text_hash: str = "", usage: dict | None = None,
    ) -> None:
        """Cache one Stage-1 extraction, keyed so a model/prompt/reasoning change
        (a different prompt_hash) naturally misses instead of serving a stale value.
        text_hash fingerprints the scored text so an enriched JD also misses. usage,
        when given, records the real provider cost/tokens for this call (cost is
        null for routes with no cost field, e.g. a direct provider API).

        Locked because the vote's worker threads write concurrently (see connect)."""
        u = usage or {}
        row = (
            listing_id, model, prompt_hash, repeat_idx, extraction_json,
            _dt(datetime.now(UTC)), text_hash,
            u.get("cost"), u.get("prompt_tokens"), u.get("completion_tokens"),
        )
        with self._extraction_lock:
            self.conn.execute(
                """
                INSERT INTO extractions
                    (listing_id, model, prompt_hash, repeat_idx, extraction, extracted_at,
                     text_hash, cost, prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(listing_id, model, prompt_hash, repeat_idx) DO UPDATE SET
                    extraction=excluded.extraction, extracted_at=excluded.extracted_at,
                    text_hash=excluded.text_hash, cost=excluded.cost,
                    prompt_tokens=excluded.prompt_tokens,
                    completion_tokens=excluded.completion_tokens
                """,
                row,
            )
            self.conn.commit()

    def get_extraction(
        self, listing_id: int, model: str, prompt_hash: str, repeat_idx: int,
        text_hash: str = "",
    ) -> str | None:
        """Return the cached extraction only when the scored text still matches
        (text_hash), so a listing whose JD was enriched is re-read, not served stale."""
        with self._extraction_lock:
            row = self.conn.execute(
                """
                SELECT extraction FROM extractions
                WHERE listing_id = ? AND model = ? AND prompt_hash = ? AND repeat_idx = ?
                  AND text_hash = ?
                """,
                (listing_id, model, prompt_hash, repeat_idx, text_hash),
            ).fetchone()
        return row["extraction"] if row else None

    # --- applications ---

    def upsert_application(self, app: Application) -> Application:
        existing = self.conn.execute(
            "SELECT id FROM applications WHERE listing_id = ?", (app.listing_id,)
        ).fetchone()

        now = _dt(datetime.now(UTC))
        if existing:
            self.conn.execute(
                """
                UPDATE applications SET status=?, applied_at=?, chase_at=?, notes=?,
                    contacts=?, notion_page_id=?, prep_content=?, updated_at=?
                WHERE listing_id=?
                """,
                (
                    app.status,
                    _dt(app.applied_at),
                    _dt(app.chase_at),
                    app.notes,
                    app.contacts,
                    app.notion_page_id,
                    app.prep_content,
                    now,
                    app.listing_id,
                ),
            )
            self.conn.commit()
            return app.model_copy(update={"id": existing["id"]})

        cur = self.conn.execute(
            """
            INSERT INTO applications
                (listing_id, status, applied_at, chase_at, notes, contacts, notion_page_id, prep_content, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                app.listing_id,
                app.status,
                _dt(app.applied_at),
                _dt(app.chase_at),
                app.notes,
                app.contacts,
                app.notion_page_id,
                app.prep_content,
                now,
            ),
        )
        self.conn.commit()
        return app.model_copy(update={"id": cur.lastrowid})

    def get_application(self, listing_id: int) -> Application | None:
        row = self.conn.execute(
            "SELECT * FROM applications WHERE listing_id = ?", (listing_id,)
        ).fetchone()
        return _row_to_application(row) if row else None

    def due_for_chase(self) -> list[dict]:
        today = datetime.now(UTC).date().isoformat()
        rows = self.conn.execute(
            """
            SELECT l.id, l.title, l.company, l.url, a.status, a.chase_at, a.notion_page_id
            FROM applications a
            JOIN listings l ON l.id = a.listing_id
            WHERE a.chase_at <= ? AND a.status NOT IN ('rejected', 'withdrawn')
            ORDER BY a.chase_at ASC
            """,
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]

    def shortlist_candidates(self, min_fit: int) -> list[dict]:
        """Listings with fit >= min_fit that don't yet have an application row."""
        rows = self.conn.execute(
            """
            SELECT l.*, s.fit_score, s.rationale, s.flags
            FROM listings l
            JOIN scores s ON s.listing_id = l.id
            WHERE s.fit_score >= ?
            AND NOT EXISTS (SELECT 1 FROM applications a WHERE a.listing_id = l.id)
            ORDER BY s.fit_score DESC
            """,
            (min_fit,),
        ).fetchall()
        return [dict(r) for r in rows]


# --- row helpers ---

def _row_to_listing(row: sqlite3.Row) -> Listing:
    d = dict(row)
    d["raw"] = json.loads(d.get("raw") or "{}")
    d["fetched_at"] = _parse_dt(d.get("fetched_at"))
    return Listing(**d)


def _row_to_score(row: sqlite3.Row) -> Score:
    d = dict(row)
    d["flags"] = json.loads(d.get("flags") or "[]")
    d["breakdown"] = json.loads(d.get("breakdown") or "{}")
    d["gate_results"] = json.loads(d.get("gate_results") or "[]")
    d["decision"] = d.get("decision") or ""
    d["tier_label"] = d.get("tier_label") or ""
    d["scored_at"] = _parse_dt(d.get("scored_at"))
    return Score(**d)


def _row_to_application(row: sqlite3.Row) -> Application:
    d = dict(row)
    d["applied_at"] = _parse_dt(d.get("applied_at"))
    d["chase_at"] = _parse_dt(d.get("chase_at"))
    d["updated_at"] = _parse_dt(d.get("updated_at"))
    return Application(**d)
