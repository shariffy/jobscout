"""Store dedup and round-trip tests (no AI, no network)."""
import os
import stat

import pytest

from jobscout.models import Application, Listing, Score
from jobscout.store import Store, _chmod_owner_only


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with Store(db) as s:
        yield s


def make_listing(**kwargs) -> Listing:
    defaults = dict(
        source_name="test-board",
        title="Senior Software Engineer",
        company="Acme Corp",
        location="Remote",
        url="https://example.com/jobs/1",
        description="We are looking for a senior engineer.",
    )
    defaults.update(kwargs)
    return Listing(**defaults)


def test_listing_round_trip(store):
    listing = make_listing()
    saved, is_new = store.upsert_listing(listing)

    assert is_new is True
    assert saved.id is not None

    fetched = store.get_listing(saved.id)
    assert fetched is not None
    assert fetched.title == listing.title
    assert fetched.company == listing.company
    assert fetched.url == listing.url
    assert fetched.description == listing.description
    assert fetched.hash == listing.hash


def test_dedup_same_listing(store):
    listing = make_listing()
    first, is_new_first = store.upsert_listing(listing)
    second, is_new_second = store.upsert_listing(listing)

    assert is_new_first is True
    assert is_new_second is False
    assert first.id == second.id


def test_dedup_different_url(store):
    a = make_listing(url="https://example.com/jobs/1", title="Role A")
    b = make_listing(url="https://example.com/jobs/2", title="Role B")

    _, new_a = store.upsert_listing(a)
    _, new_b = store.upsert_listing(b)

    assert new_a is True
    assert new_b is True


def test_hash_stable_across_instances():
    a = make_listing()
    b = make_listing()
    assert a.hash == b.hash


def test_score_round_trip(store):
    listing, _ = store.upsert_listing(make_listing())

    score = Score(
        listing_id=listing.id,
        rationale="Strong match on Python and product experience.",
        flags=["remote-friendly", "series-b"],
        model="claude-haiku-4-5",
        tier="bulk",
        decision="apply",
        priority=12,
        tier_label="T1",
    )
    saved = store.insert_score(score)
    assert saved.id is not None

    best = store.get_best_score(listing.id)
    assert best is not None
    assert best.decision == "apply"
    assert best.priority == 12
    assert "remote-friendly" in best.flags


def test_unscored_listings(store):
    a, _ = store.upsert_listing(make_listing(url="https://example.com/jobs/1", title="Role A"))
    b, _ = store.upsert_listing(make_listing(url="https://example.com/jobs/2", title="Role B"))

    store.insert_score(Score(
        listing_id=a.id, rationale="ok", model="haiku", tier="bulk", decision="apply",
    ))

    unscored = store.unscored_listings()
    assert len(unscored) == 1
    assert unscored[0].id == b.id


def test_application_upsert(store):
    listing, _ = store.upsert_listing(make_listing())
    app = Application(listing_id=listing.id, status="applied", notes="Applied via referral.")
    saved = store.upsert_application(app)
    assert saved.id is not None

    fetched = store.get_application(listing.id)
    assert fetched is not None
    assert fetched.status == "applied"
    assert fetched.notes == "Applied via referral."

    updated = Application(listing_id=listing.id, status="interviewing", notes="Phone screen booked.")
    store.upsert_application(updated)
    refetched = store.get_application(listing.id)
    assert refetched.status == "interviewing"


def test_list_listings_apply_only(store):
    a, _ = store.upsert_listing(make_listing(url="https://example.com/1", title="Reject"))
    b, _ = store.upsert_listing(make_listing(url="https://example.com/2", title="Apply me"))

    store.insert_score(Score(
        listing_id=a.id, rationale="weak", model="haiku", tier="bulk", decision="no",
    ))
    store.insert_score(Score(
        listing_id=b.id, rationale="strong", model="haiku", tier="bulk",
        decision="apply", priority=10, tier_label="T1",
    ))

    results = store.list_listings(apply_only=True)
    assert len(results) == 1
    assert results[0]["title"] == "Apply me"


def test_gated_score_round_trip(store):
    listing, _ = store.upsert_listing(make_listing())
    gate_results = [{"name": "office", "status": "pass", "reason": "2 <= 3",
                     "evidence": "2 days/week", "confidence": 1.0}]
    score = Score(
        listing_id=listing.id, rationale="APPLY T1", flags=[],
        model="test/model", tier="bulk",
        decision="apply", priority=2, tier_label="T1", gate_results=gate_results,
    )
    store.insert_score(score)

    best = store.get_best_score(listing.id)
    assert best.decision == "apply"
    assert best.priority == 2
    assert best.tier_label == "T1"
    assert best.gate_results == gate_results


def test_insert_score_defaults_empty_decision_to_no(store):
    listing, _ = store.upsert_listing(make_listing())
    saved = store.insert_score(Score(
        listing_id=listing.id, rationale="", model="m", tier="bulk",
    ))
    assert saved.decision == "no"


def test_migration_drops_fit_and_backfills_decision(tmp_path):
    """Opening a pre-gated DB drops fit_score, backfills decision, dedupes."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, hash TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL, title TEXT NOT NULL, company TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT '', url TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '', raw TEXT NOT NULL DEFAULT '{}',
            fetched_at TEXT NOT NULL
        );
        CREATE TABLE scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL REFERENCES listings(id),
            fit_score INTEGER NOT NULL, rationale TEXT NOT NULL DEFAULT '',
            flags TEXT NOT NULL DEFAULT '[]', breakdown TEXT NOT NULL DEFAULT '{}',
            model TEXT NOT NULL, tier TEXT NOT NULL, scored_at TEXT NOT NULL
        );
        INSERT INTO listings (hash, source_name, title, company, url, fetched_at)
            VALUES ('h1', 's', 'Role', 'Co', 'https://x.test/1', '2026-01-01T00:00:00+00:00');
        INSERT INTO scores (listing_id, fit_score, model, tier, scored_at)
            VALUES (1, 55, 'm', 'bulk', '2026-01-01T00:00:00+00:00'),
                   (1, 85, 'm', 'bulk', '2026-01-02T00:00:00+00:00');
    """)
    conn.commit()
    conn.close()

    with Store(db) as s:
        cols = {r["name"] for r in s.conn.execute("PRAGMA table_info(scores)")}
        assert "fit_score" not in cols
        rows = s.conn.execute("SELECT decision, rationale FROM scores").fetchall()
        assert len(rows) == 1
        assert rows[0]["decision"] == "apply"
        best = s.get_best_score(1)
        assert best.decision == "apply"
        assert best.priority is None
        assert best.gate_results == []


def test_migration_adds_cascade_and_unique_constraint(tmp_path):
    """A pre-existing DB (no CASCADE, no UNIQUE on scores.listing_id) is rebuilt
    to have both on open."""
    import sqlite3

    db = tmp_path / "old2.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, hash TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL, title TEXT NOT NULL, company TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT '', url TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '', raw TEXT NOT NULL DEFAULT '{}',
            fetched_at TEXT NOT NULL
        );
        CREATE TABLE scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL REFERENCES listings(id),
            fit_score INTEGER NOT NULL, rationale TEXT NOT NULL DEFAULT '',
            flags TEXT NOT NULL DEFAULT '[]', breakdown TEXT NOT NULL DEFAULT '{}',
            model TEXT NOT NULL, tier TEXT NOT NULL, scored_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

    with Store(db) as s:
        fks = s.conn.execute("PRAGMA foreign_key_list(scores)").fetchall()
        assert any(fk["table"] == "listings" and fk["on_delete"] == "CASCADE" for fk in fks)
        indexes = s.conn.execute("PRAGMA index_list(scores)").fetchall()
        assert any(idx["unique"] for idx in indexes)
        cols = {r["name"] for r in s.conn.execute("PRAGMA table_info(scores)")}
        assert "fit_score" not in cols


def test_score_cascade_deletes_with_listing(store):
    listing, _ = store.upsert_listing(make_listing())
    store.insert_score(Score(
        listing_id=listing.id, rationale="", model="m", tier="bulk", decision="apply",
    ))

    store.conn.execute("DELETE FROM listings WHERE id = ?", (listing.id,))
    store.conn.commit()

    assert store.get_best_score(listing.id) is None
    row = store.conn.execute(
        "SELECT COUNT(*) c FROM scores WHERE listing_id = ?", (listing.id,)
    ).fetchone()
    assert row["c"] == 0


def test_insert_score_upserts_one_row_per_listing(store):
    listing, _ = store.upsert_listing(make_listing())
    store.insert_score(Score(
        listing_id=listing.id, rationale="first", model="m", tier="bulk", decision="no",
    ))
    store.insert_score(Score(
        listing_id=listing.id, rationale="second", model="m", tier="bulk",
        decision="apply", priority=5, tier_label="T2",
    ))

    rows = store.conn.execute(
        "SELECT decision, rationale FROM scores WHERE listing_id = ?", (listing.id,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["decision"] == "apply"
    assert rows[0]["rationale"] == "second"


def test_upsert_listing_refreshes_richer_description(store):
    a, is_new = store.upsert_listing(make_listing(description="short"))
    assert is_new is True

    b, is_new2 = store.upsert_listing(make_listing(description="a much longer and richer description"))
    assert is_new2 is False
    assert b.id == a.id

    fetched = store.get_listing(a.id)
    assert fetched.description == "a much longer and richer description"


def test_upsert_listing_keeps_existing_when_new_description_shorter(store):
    store.upsert_listing(make_listing(description="a much longer and richer description"))
    store.upsert_listing(make_listing(description="short"))

    fetched = store.get_listing(1)
    assert fetched.description == "a much longer and richer description"


def test_update_listing_description(store):
    listing, _ = store.upsert_listing(make_listing(description="short"))
    store.update_listing_description(listing.id, "enriched full JD text")
    assert store.get_listing(listing.id).description == "enriched full JD text"


def test_score_round_trip_includes_assessment_version(store):
    listing, _ = store.upsert_listing(make_listing())
    store.insert_score(Score(
        listing_id=listing.id, rationale="", model="m", tier="bulk",
        decision="apply", assessment_version="abc123",
    ))
    best = store.get_best_score(listing.id)
    assert best.assessment_version == "abc123"


def test_extraction_cache_round_trip(store):
    listing, _ = store.upsert_listing(make_listing())
    assert store.get_extraction(listing.id, "model-a", "hash-a", 0) is None

    store.save_extraction(listing.id, "model-a", "hash-a", 0, '{"features": {}, "rules": {}}')
    cached = store.get_extraction(listing.id, "model-a", "hash-a", 0)
    assert cached == '{"features": {}, "rules": {}}'

    assert store.get_extraction(listing.id, "model-b", "hash-a", 0) is None
    assert store.get_extraction(listing.id, "model-a", "hash-a", 1) is None


def test_extraction_cache_overwrite(store):
    listing, _ = store.upsert_listing(make_listing())
    store.save_extraction(listing.id, "model-a", "hash-a", 0, '{"v": 1}')
    store.save_extraction(listing.id, "model-a", "hash-a", 0, '{"v": 2}')
    assert store.get_extraction(listing.id, "model-a", "hash-a", 0) == '{"v": 2}'


def test_extraction_cascade_deletes_with_listing(store):
    listing, _ = store.upsert_listing(make_listing())
    store.save_extraction(listing.id, "model-a", "hash-a", 0, '{"v": 1}')

    store.conn.execute("DELETE FROM listings WHERE id = ?", (listing.id,))
    store.conn.commit()

    row = store.conn.execute(
        "SELECT COUNT(*) c FROM extractions WHERE listing_id = ?", (listing.id,)
    ).fetchone()
    assert row["c"] == 0


@pytest.mark.skipif(os.name != "posix", reason="chmod perms are posix-only")
def test_new_db_is_created_owner_only(tmp_path):
    db = tmp_path / "perms.db"
    with Store(db):
        pass
    mode = stat.S_IMODE(db.stat().st_mode)
    assert mode == 0o600


@pytest.mark.skipif(os.name != "posix", reason="chmod perms are posix-only")
def test_chmod_owner_only_restricts_existing_file(tmp_path):
    f = tmp_path / "secret.txt"
    f.write_text("shh")
    f.chmod(0o644)

    _chmod_owner_only(f)

    assert stat.S_IMODE(f.stat().st_mode) == 0o600
