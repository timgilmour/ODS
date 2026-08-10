"""Tests for app.actuation — the one process-wide actuation lock shared by
tick(), set apply, and the pull-through completion hook (max-review #6/#7).

Thin by design: the module itself is nine lines. The interesting behavior —
tick's try-acquire, apply's blocking acquire, the pull-through hook's
supersession check — is covered in tests/test_arbiter.py and
tests/test_storage_api.py, against the real actors. This file only proves
the primitive itself: a plain lock, and a peek that never acquires.
"""

import threading

from app import actuation


def test_in_progress_false_when_unheld():
    assert actuation.in_progress() is False


def test_in_progress_true_while_locked():
    with actuation.LOCK:
        assert actuation.in_progress() is True
    assert actuation.in_progress() is False


def test_in_progress_is_a_peek_not_an_acquire():
    """in_progress() must never itself hold the lock — a second thread
    acquiring it right after a True peek must succeed immediately, not
    block. Bounded wait: a deadlock fails the test, not the suite."""
    with actuation.LOCK:
        assert actuation.in_progress() is True

    acquired = threading.Event()

    def _grab():
        with actuation.LOCK:
            acquired.set()

    t = threading.Thread(target=_grab)
    t.start()
    assert acquired.wait(timeout=5), "in_progress() left the lock held"
    t.join(timeout=5)
    assert not t.is_alive()
