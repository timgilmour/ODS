"""
Model Deck event log — append-only JSONL audit trail.

``log_event()`` appends one JSON line per call, and ROTATES: once the file
passes ``_MAX_LOG_BYTES`` it is rewritten to its newest ``_TRIM_TO_LINES``
lines. The log is display-only and ``tail_events`` reads the whole file to
return its last n, so unbounded growth cost a growing read on every Events
poll [max-review c7]. Old lines are expendable here in a way a state file's
contents never are.

Each call writes:
``{"ts": "<UTC ISO8601 with timezone>", "kind": kind, "detail": detail}``.
Parent directories are created on first write, matching the idiom used by
``app.registry``/``app.policy`` for their JSON stores. Writes are plain
appends (not atomic-replace like those two) since this is a log, not a
point-in-time record — losing the tail of a single in-flight write on a
crash is an acceptable risk for a display-only audit trail.

``tail_events()`` reads the last ``n`` entries back, oldest -> newest,
silently skipping any line that doesn't parse as JSON. A missing file reads
as an empty list rather than raising, the same "self-heals, never blocks
the caller" quality bar ``app.registry``/``app.policy`` use for corrupt
state files.

No Settings import here — pure inputs only.
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

# ONE process-wide lock for the append+rotate pair. log_event is called from
# essentially every thread the deck runs — the arbiter watcher, the storage
# watcher, the mover, the update-checker, and all ~90 sync HTTP routes (which
# FastAPI runs on a real threadpool) — and _trim writes a FIXED tmp path per
# log. Two concurrent trims race that one path, and the loser's os.replace
# raises FileNotFoundError into whatever thread it was: a 500 on an HTTP
# route, or a dead pass on a watcher. Module-level rather than per-store
# because callers pass a Path, not an object, so there is no instance to hang
# it on; contention is negligible (one small append per event).
_WRITE_LOCK = threading.Lock()

# Rotation bounds. The log is display-only (the Events tab reads the tail).
#
# The TRIGGER is bytes and the TARGET must be bytes too, or there is no
# hysteresis: a line-only target gets back under a byte trigger only while
# the trailing window happens to average below _MAX_LOG_BYTES/_TRIM_TO_LINES
# (2500 B/line). Above that ratio the file stays oversized after every trim,
# so EVERY subsequent append rewrites the whole multi-MB log — sustained
# thrashing rather than periodic amortization. That is reachable, not
# theoretical: engines raise EngineError(resp.text) carrying a raw HTTP body
# (app/engines/lemonade.py, app/engines/spark.py), and app/arbiter.py logs
# `load-failed` with str(exc) UN-deduped on every 2 s tick, so one degraded
# backend returning a multi-KB error body thrashes for the whole outage.
#
# _TRIM_TARGET_BYTES is well under the trigger so a trim buys real headroom
# whatever the record size. _TRIM_TO_LINES still caps the count (~20x what
# the UI ever requests) for the ordinary small-record case.
_MAX_LOG_BYTES = 5_000_000
_TRIM_TARGET_BYTES = 2_000_000
_TRIM_TO_LINES = 2_000


def log_event(path: Path, kind: str, detail: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(UTC).isoformat(), "kind": kind, "detail": detail}
    with _WRITE_LOCK:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        _rotate_if_oversized(path)


def _rotate_if_oversized(path: Path) -> None:
    """Caller holds _WRITE_LOCK."""
    try:
        oversized = path.stat().st_size > _MAX_LOG_BYTES
    except OSError:
        # The append above succeeded; a stat that fails is not worth failing
        # a caller's actuation over. Next call re-checks.
        return
    if oversized:
        _trim(path)


def _trim(path: Path) -> None:
    """Rewrite the log to its newest ``_TRIM_TO_LINES`` lines.

    Caller holds _WRITE_LOCK. Atomic replace, the same idiom the JSON stores
    use, so a reader never sees a half-written log.

    The lock is not about losing a few tail lines — the real failure without
    it is a CRASH: this writes one fixed ``.trim-tmp`` path per log, so two
    concurrent trims race it and the loser's ``os.replace`` raises
    FileNotFoundError into whichever thread it was (a 500 on an HTTP route, a
    dead watcher pass). Rare, because it is gated behind the size threshold —
    rare is not safe.
    """
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    lines = lines[-_TRIM_TO_LINES:]
    # ...then trim again by BYTES, newest-first, so the rewritten file is
    # actually under the target no matter how large the records are.
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        total += len(line) + 1          # +1 for the newline it is joined with
        if total > _TRIM_TARGET_BYTES and kept:
            break
        kept.append(line)
    lines = list(reversed(kept))
    tmp = path.with_suffix(path.suffix + ".trim-tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def tail_events(path: Path, n: int = 100) -> list[dict]:
    try:
        text = path.read_text()
    except OSError:
        return []

    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-n:]
