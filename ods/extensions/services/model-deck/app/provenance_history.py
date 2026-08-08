"""Durable provenance history — append-only JSONL, one line per transition.

WHY THIS IS NOT app.events. That log's own docstring calls it a
"display-only audit trail" that accepts "losing the tail of a single
in-flight write on a crash", deliberately not atomic. History has to answer
"put ds4 back to what it was last Tuesday" (design decision D6), so it gets
its own file and three guarantees events.py does not make:

1. ``fsync`` before returning — a crash after the call cannot lose the record.
2. **Newline repair.** If the file does not end in a newline, one is written
   first. Without it a torn line silently concatenates with its successor and
   destroys TWO records instead of one.
3. Its own file, so a noisy UI event stream cannot interleave with it.

Append-only is still safe for AUTHORITATIVE data here, unlike events.py,
because it is not the authority: current state lives in provenance.json,
written atomically. This file records how it got there. The worst case of a
torn write is therefore a gap in the narrative — never a wrong current state
and never an unrestorable system.

``to`` embeds the WHOLE block rather than a delta. That is what makes
state_at a single scan instead of a replay, and what keeps one lost line
from invalidating everything after it.

History is never rewritten, compacted, or truncated.
"""

import json
import os
from pathlib import Path

FIELDS = ("current", "desired", "origin")
CAUSES = ("observed", "declared", "adopted", "backfill")
ACTORS = ("watcher", "api")


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record)
    needs_newline = _needs_newline(path)
    with path.open("a") as f:
        if needs_newline:
            f.write("\n")
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _needs_newline(path: Path) -> bool:
    """True when a previous write was torn mid-line (see the module
    docstring, guarantee 2). Reads one byte, not the file."""
    try:
        size = path.stat().st_size
        if size == 0:
            return False
        with path.open("rb") as f:
            f.seek(size - 1)
            return f.read(1) != b"\n"
    except OSError:
        return False


def read_all(path: Path) -> list[dict]:
    """Every parseable record, oldest first. A missing file is empty, and an
    unparseable line is skipped — app.events.tail_events' quality bar, for
    the same reason: a corrupt line must never block the reader."""
    try:
        text = path.read_text()
    except OSError:
        return []
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def history_for(path: Path, artifact_id: str) -> list[dict]:
    return [r for r in read_all(path) if r.get("artifact_id") == artifact_id]


def state_at(path: Path, artifact_id: str, ts: str,
             field: str = "current") -> dict | None:
    """The `field` block in force for `artifact_id` at `ts` — the last
    transition at or before it, or None if the artifact had no such block
    yet. ISO-8601 timestamps with a fixed offset sort correctly as plain
    strings, the same assumption node-agent's read_newest_catalog makes."""
    latest = None
    for record in history_for(path, artifact_id):
        if record.get("field") != field:
            continue
        if record.get("ts", "") <= ts:
            latest = record
    return None if latest is None else latest.get("to")
