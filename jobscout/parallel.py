"""Bounded concurrency for per-listing work (scoring, JD fetches).

One helper, `map_bounded`, runs a `work(item)` callable over many items with at most
`max_workers` in flight, and delivers each result to `on_result` **on the calling
thread** as it completes. That split is the whole point: `work` runs on a worker
thread and may only touch thread-safe resources (the OpenRouter client, httpx, the
extraction cache behind Store's lock), while `on_result` runs serially on the caller's
thread and may freely touch things that are NOT thread-safe — the sqlite Store's other
tables, the Rich console, the Notion client.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# on_result signature: (done, total, item, result, error) — result is None on
# failure, error is None on success.
OnResult = Callable[[int, int, Any, Any, "Exception | None"], None]


def map_bounded(
    items: Iterable[Any],
    work: Callable[[Any], Any],
    on_result: OnResult,
    *,
    max_workers: int,
) -> None:
    """Run work(item) over items, <= max_workers concurrently, calling on_result on
    THIS thread as each finishes. Per-item exceptions are caught and passed to
    on_result (result=None, error=exc) so one failure never sinks the batch.

    max_workers <= 1 (or a single item) runs serially with no threads, preserving
    exact ordering and behaviour for the degenerate case.
    """
    items = list(items)
    total = len(items)
    if total == 0:
        return
    if max_workers <= 1 or total == 1:
        for i, item in enumerate(items, 1):
            _deliver(on_result, i, total, item, work)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(work, item): item for item in items}
        for done, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 — reported to on_result, not swallowed
                on_result(done, total, item, None, exc)
            else:
                on_result(done, total, item, result, None)


def _deliver(
    on_result: OnResult, i: int, total: int, item: Any, work: Callable[[Any], Any]
) -> None:
    try:
        result = work(item)
    except Exception as exc:  # noqa: BLE001 — reported to on_result, not swallowed
        on_result(i, total, item, None, exc)
    else:
        on_result(i, total, item, result, None)


def score_batch_parallel(scorer, listings, *, max_workers, progress_cb=None):
    """Parallel drop-in for a serial score loop: score every listing (<= max_workers
    at once) and return the Scores in INPUT order. progress_cb(done, total, listing,
    score), when given, fires on the calling thread as each finishes (completion
    order). Unlike the CLI paths, this preserves the old contract of propagating the
    first scoring error rather than reporting it per-item.
    """
    listings = list(listings)
    results: dict = {}

    def on_result(done, total, indexed, score, error):
        idx, listing = indexed
        if error is not None:
            raise error
        results[idx] = score
        if progress_cb:
            progress_cb(done, total, listing, score)

    map_bounded(
        list(enumerate(listings)),
        lambda pair: scorer.score(pair[1]),
        on_result,
        max_workers=max_workers,
    )
    return [results[i] for i in range(len(listings))]
