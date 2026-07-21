"""
Model Deck event log — append-only JSONL audit trail.

``log_event()`` appends one JSON line per call:
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
from datetime import UTC, datetime
from pathlib import Path


def log_event(path: Path, kind: str, detail: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(UTC).isoformat(), "kind": kind, "detail": detail}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


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
