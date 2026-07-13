"""_apply_status tests — the shared helper behind `set-status` and watch's
board-drift sync (jobscout.cli._sync_status_drift is tested separately in
test_watch_sync.py, which exercises this same helper end-to-end)."""
from unittest.mock import MagicMock

import pytest

from jobscout.cli import _apply_status
from jobscout.models import Listing
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


def test_missing_listing_returns_none(store):
    listing, saved, err = _apply_status(store, None, 999, "applied")
    assert listing is None
    assert saved is None
    assert err is None


def test_sets_status_locally_with_no_notion(store):
    listing, _ = store.upsert_listing(make_listing())
    out_listing, saved, err = _apply_status(store, None, listing.id, "rejected")
    assert out_listing.id == listing.id
    assert saved.status == "rejected"
    assert err is None


def test_applied_computes_chase_date(store):
    listing, _ = store.upsert_listing(make_listing())
    _, saved, _ = _apply_status(store, None, listing.id, "applied", chase_days=10)
    assert saved.applied_at is not None
    assert saved.chase_at is not None
    assert (saved.chase_at - saved.applied_at).days == 10


def test_non_applied_status_leaves_chase_unset(store):
    listing, _ = store.upsert_listing(make_listing())
    _, saved, _ = _apply_status(store, None, listing.id, "shortlisted")
    assert saved.applied_at is None
    assert saved.chase_at is None


def test_no_notion_client_skips_push(store):
    from jobscout.models import Application

    listing, _ = store.upsert_listing(make_listing())
    store.upsert_application(Application(listing_id=listing.id, notion_page_id="p1"))
    _, saved, err = _apply_status(store, None, listing.id, "rejected")
    assert saved.status == "rejected"
    assert err is None  # nothing to report — there's no ns to fail against


def test_pushes_to_notion_and_archives_on_terminal_status(store):
    from jobscout.models import Application

    listing, _ = store.upsert_listing(make_listing())
    store.upsert_application(Application(listing_id=listing.id, notion_page_id="p1"))
    ns = MagicMock()

    _apply_status(store, ns, listing.id, "not_interested")

    ns.update_status.assert_called_once()
    assert ns.update_status.call_args.args[0] == "p1"
    ns.archive_page.assert_called_once_with("p1")


def test_applied_does_not_archive(store):
    from jobscout.models import Application

    listing, _ = store.upsert_listing(make_listing())
    store.upsert_application(Application(listing_id=listing.id, notion_page_id="p1"))
    ns = MagicMock()

    _apply_status(store, ns, listing.id, "applied")

    ns.update_status.assert_called_once()
    ns.archive_page.assert_not_called()


def test_notion_error_is_reported_not_raised(store):
    from jobscout.models import Application

    listing, _ = store.upsert_listing(make_listing())
    store.upsert_application(Application(listing_id=listing.id, notion_page_id="p1"))
    ns = MagicMock()
    ns.update_status.side_effect = RuntimeError("boom")

    out_listing, saved, err = _apply_status(store, ns, listing.id, "rejected")

    # local write must still succeed even though the Notion push failed
    assert saved.status == "rejected"
    assert err == "boom"


def test_notion_page_id_links_an_unlinked_local_row(store):
    """Board-drift sync passes notion_page_id explicitly so a listing with no
    prior Application row (or one missing notion_page_id) still gets synced
    back to the page it was actually found on."""
    listing, _ = store.upsert_listing(make_listing())
    ns = MagicMock()

    _, saved, err = _apply_status(
        store, ns, listing.id, "interviewing", notion_page_id="new-page-id"
    )

    assert saved.notion_page_id == "new-page-id"
    ns.update_status.assert_called_once()
    assert ns.update_status.call_args.args[0] == "new-page-id"
