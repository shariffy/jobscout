"""Tests for jobscout.parallel — bounded concurrency helpers."""
import threading
import time

from jobscout.parallel import map_bounded


def test_map_bounded_delivers_all_results():
    seen = {}
    map_bounded(range(5), lambda n: n * n, lambda d, t, i, r, e: seen.__setitem__(i, r),
                max_workers=3)
    assert seen == {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}


def test_map_bounded_reports_errors_not_raises():
    results, errors = {}, {}

    def work(n):
        if n == 2:
            raise ValueError("boom")
        return n

    def on_result(done, total, item, result, error):
        (errors if error else results)[item] = error or result

    map_bounded(range(4), work, on_result, max_workers=2)
    assert results == {0: 0, 1: 1, 3: 3}
    assert isinstance(errors[2], ValueError)


def test_map_bounded_on_result_runs_on_calling_thread():
    # The whole safety argument: on_result must run on the caller's thread, never a
    # worker — so it may touch non-thread-safe resources (store, console).
    caller = threading.get_ident()
    worker_threads, result_threads = set(), set()

    def work(n):
        worker_threads.add(threading.get_ident())
        return n

    def on_result(done, total, item, result, error):
        result_threads.add(threading.get_ident())

    map_bounded(range(6), work, on_result, max_workers=4)
    assert result_threads == {caller}
    assert worker_threads - {caller}  # work actually ran off-thread


def test_map_bounded_runs_concurrently():
    # 4 items each sleeping 0.2s finish in ~0.2s with 4 workers, not ~0.8s serial.
    start = time.perf_counter()
    map_bounded(range(4), lambda n: time.sleep(0.2), lambda *a: None, max_workers=4)
    assert time.perf_counter() - start < 0.5


def test_map_bounded_serial_when_single_worker():
    order = []
    map_bounded(range(4), lambda n: order.append(("work", n)) or n,
                lambda d, t, i, r, e: order.append(("done", i)), max_workers=1)
    # serial => each item's work is immediately followed by its delivery, in order
    assert order == [("work", 0), ("done", 0), ("work", 1), ("done", 1),
                     ("work", 2), ("done", 2), ("work", 3), ("done", 3)]


def test_map_bounded_empty_is_noop():
    calls = []
    map_bounded([], lambda n: n, lambda *a: calls.append(a), max_workers=4)
    assert calls == []


