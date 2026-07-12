from __future__ import annotations

import json
import sqlite3
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
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id   INTEGER NOT NULL REFERENCES listings(id),
    fit_score    INTEGER NOT NULL,
    rationale    TEXT    NOT NULL DEFAULT '',
    flags        TEXT    NOT NULL DEFAULT '[]',
    breakdown    TEXT    NOT NULL DEFAULT '{}',
    model        TEXT    NOT NULL,
    tier         TEXT    NOT NULL,
    decision     TEXT    NOT NULL DEFAULT '',
    priority     INTEGER,
    tier_label   TEXT    NOT NULL DEFAULT '',
    gate_results TEXT    NOT NULL DEFAULT '[]',
    scored_at    TEXT    NOT NULL
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

    def connect(self) -> None:
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
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
        ]:
            if column not in existing:
                self.conn.execute(f"ALTER TABLE scores ADD COLUMN {ddl}")
        self.conn.execute(
            "UPDATE scores SET decision = CASE WHEN fit_score >= 70 THEN 'apply' ELSE 'no' END "
            "WHERE decision = ''"
        )

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
        """Insert listing; return (listing_with_id, is_new). Dedups on hash."""
        row = self.conn.execute(
            "SELECT id FROM listings WHERE hash = ?", (listing.hash,)
        ).fetchone()

        if row:
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

    def list_listings(self, min_fit: int | None = None, limit: int = 100) -> list[dict]:
        """Return listings joined with their best score, optionally filtered by fit."""
        sql = """
            SELECT l.*, s.fit_score, s.rationale, s.flags, s.tier,
                   s.decision, s.priority, s.tier_label
            FROM listings l
            LEFT JOIN (
                SELECT listing_id, MAX(fit_score) AS fit_score, rationale, flags, tier,
                       decision, priority, tier_label
                FROM scores
                GROUP BY listing_id
            ) s ON s.listing_id = l.id
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
        # Legacy additive scores carry no decision; store the fit-derived one so
        # decision-based queries see every row.
        decision = score.decision or ("apply" if score.fit_score >= 70 else "no")
        cur = self.conn.execute(
            """
            INSERT INTO scores (listing_id, fit_score, rationale, flags, breakdown, model, tier,
                                decision, priority, tier_label, gate_results, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _dt(score.scored_at),
            ),
        )
        self.conn.commit()
        return score.model_copy(update={"id": cur.lastrowid, "decision": decision})

    def get_best_score(self, listing_id: int) -> Score | None:
        row = self.conn.execute(
            "SELECT * FROM scores WHERE listing_id = ? ORDER BY fit_score DESC LIMIT 1",
            (listing_id,),
        ).fetchone()
        return _row_to_score(row) if row else None

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
        """Listings with best fit >= min_fit that don't yet have an application row."""
        rows = self.conn.execute(
            """
            SELECT l.*, s.fit_score, s.rationale, s.flags
            FROM listings l
            JOIN (
                SELECT listing_id, MAX(fit_score) AS fit_score, rationale, flags
                FROM scores GROUP BY listing_id
            ) s ON s.listing_id = l.id
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
