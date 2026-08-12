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
"the block in force at time T" a single scan instead of a replay, and what
keeps one lost line from invalidating everything after it. (A ``state_at``
reader implementing that scan existed but had no caller anywhere — deleted
in the 2026-08-12 simplify sweep, recoverable from git when a D6
restore-point route is actually built.)

History is never rewritten, compacted, or truncated.
"""

import json
import os
from pathlib import Path

# THE VOCABULARY THIS FILE RECORDS. Documentation, not enforcement: nothing
# reads these tuples (grep-verified), `append` writes whatever it is handed,
# and a reader must tolerate a value from a future version anyway. They exist
# so the one file that IS the record also says what the record can contain --
# which means a producer added without updating them makes this file lie.
# tests/test_provenance_history.py parses ProvenanceStore._record's call
# sites and asserts these three cover them exactly, so the lie cannot last.
FIELDS = ("current", "desired", "origin", "watch", "update.status")
CAUSES = ("observed", "declared", "checked")
ACTORS = ("watcher", "api", "operator", "update-checker")


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
