"""Status-drift sync tests for watch() — pure logic, no real Notion calls.

Covers _sync_status_drift: pulling a manual Status edit made directly on the
Notion board (rather than through an Action) back into SQLite.
"""
from unittest.mock import MagicMock

import pytest

from jobscout.cli import _sync_status_drift
from jobscout.models import Application, Listing
from jobscout.store import Store


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with Store(db) as s:
        yield s


def make_listing(**kwargs) -> Listing:
    defaults = dict(
        source_name="test-board", title="Senior Engineer", company="Acme",
        url="https://example.com/jobs/1",
    )
    defaults.update(kwargs)
    return Listing(**defaults)


def seed(store, status="shortlisted", page_id="p1") -> Listing:
    listing, _ = store.upsert_listing(make_listing())
    store.upsert_application(
        Application(listing_id=listing.id, status=status, notion_page_id=page_id)
    )
    return listing


def test_no_drift_is_a_noop(store):
    listing = seed(store)
    ns = MagicMock()

    _sync_status_drift(store, ns, MagicMock(), [
        {"page_id": "p1", "listing_id": listing.id, "status": "shortlisted", "notes": ""}
    ])

    assert store.get_application(listing.id).status == "shortlisted"
    ns.archive_page.assert_not_called()
    ns.update_status.assert_not_called()


def test_manual_status_change_syncs_locally(store):
    listing = seed(store)
    ns = MagicMock()

    _sync_status_drift(store, ns, MagicMock(), [
        {"page_id": "p1", "listing_id": listing.id, "status": "interviewing", "notes": ""}
    ])

    assert store.get_application(listing.id).status == "interviewing"
    ns.archive_page.assert_not_called()


def test_manual_applied_computes_chase_date(store):
    listing = seed(store)
    ns = MagicMock()

    _sync_status_drift(store, ns, MagicMock(), [
        {"page_id": "p1", "listing_id": listing.id, "status": "applied", "notes": ""}
    ])

    app = store.get_application(listing.id)
    assert app.status == "applied"
    assert app.applied_at is not None
    assert app.chase_at is not None
    assert (app.chase_at - app.applied_at).days == 7
    ns.update_status.assert_called_once()
    ns.archive_page.assert_not_called()


@pytest.mark.parametrize("status", ["not_interested", "rejected", "withdrawn"])
def test_manual_terminal_status_archives(store, status):
    listing = seed(store)
    ns = MagicMock()

    _sync_status_drift(store, ns, MagicMock(), [
        {"page_id": "p1", "listing_id": listing.id, "status": status, "notes": ""}
    ])

    assert store.get_application(listing.id).status == status
    ns.archive_page.assert_called_once_with("p1")


def test_notes_carried_over_on_sync(store):
    listing = seed(store)
    ns = MagicMock()

    _sync_status_drift(store, ns, MagicMock(), [
        {"page_id": "p1", "listing_id": listing.id, "status": "rejected",
         "notes": "phone screen didn't go well"}
    ])

    assert store.get_application(listing.id).notes == "phone screen didn't go well"


def test_missing_listing_is_skipped(store):
    ns = MagicMock()
    _sync_status_drift(store, ns, MagicMock(), [
        {"page_id": "p1", "listing_id": 999, "status": "rejected", "notes": ""}
    ])
    ns.archive_page.assert_not_called()
