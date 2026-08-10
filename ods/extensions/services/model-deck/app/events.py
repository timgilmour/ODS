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
from datetime import UTC, datetime
from pathlib import Path

# Rotation bounds. The log is display-only (the Events tab reads the tail);
# 2000 lines is ~20x what the UI ever requests, and the byte cap keeps one
# pathological detail payload from deferring rotation indefinitely.
_MAX_LOG_BYTES = 5_000_000
_TRIM_TO_LINES = 2_000


def log_event(path: Path, kind: str, detail: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(UTC).isoformat(), "kind": kind, "detail": detail}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
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

    Atomic replace, the same idiom the JSON stores use, so a reader never
    sees a half-written log. Two racing trims can drop a handful of tail
    lines — acceptable at this module's stated quality bar (a display-only
    audit trail whose appends are already lossy on a crash), and not
    acceptable anywhere state is kept.
    """
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    lines = lines[-_TRIM_TO_LINES:]
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
