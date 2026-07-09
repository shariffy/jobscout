"""Store dedup and round-trip tests (no AI, no network)."""
import pytest

from jobscout.models import Application, Listing, Score
from jobscout.store import Store


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
        fit_score=82,
        rationale="Strong match on Python and product experience.",
        flags=["remote-friendly", "series-b"],
        model="claude-haiku-4-5",
        tier="bulk",
    )
    saved = store.insert_score(score)
    assert saved.id is not None

    best = store.get_best_score(listing.id)
    assert best is not None
    assert best.fit_score == 82
    assert "remote-friendly" in best.flags


def test_unscored_listings(store):
    a, _ = store.upsert_listing(make_listing(url="https://example.com/jobs/1", title="Role A"))
    b, _ = store.upsert_listing(make_listing(url="https://example.com/jobs/2", title="Role B"))

    store.insert_score(Score(
        listing_id=a.id, fit_score=75, rationale="ok", model="haiku", tier="bulk"
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


def test_list_listings_min_fit(store):
    a, _ = store.upsert_listing(make_listing(url="https://example.com/1", title="Low fit"))
    b, _ = store.upsert_listing(make_listing(url="https://example.com/2", title="High fit"))

    store.insert_score(Score(listing_id=a.id, fit_score=40, rationale="weak", model="haiku", tier="bulk"))
    store.insert_score(Score(listing_id=b.id, fit_score=85, rationale="strong", model="haiku", tier="bulk"))

    results = store.list_listings(min_fit=70)
    assert len(results) == 1
    assert results[0]["title"] == "High fit"
