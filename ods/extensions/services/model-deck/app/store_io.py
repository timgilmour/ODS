"""One home for the JSON-store file idiom [max-review c8].

Twelve store classes hand-copied the same two halves — read
(``read_text`` → ``json.loads`` → shape heal) and atomic write (``mkdir``
→ ``.tmp`` sibling → ``os.replace``) — and the copies had measurably
drifted (some sort_keys, some not; sets.py:158 even carried a comment
"same temp+os.replace idiom as registry/policy"). A durability fix to
either half (fsync-before-replace, distinct OSError handling) previously
had to be re-applied per store and would predictably miss some; now it
lands here once.

What deliberately does NOT live here:

* Each store's HEAL. The empty shape and any deeper repair differ per
  store on purpose — policy heals per-kind to defaults, intent DROPS
  malformed records, settings heals recursively. ``load_json`` only
  answers "missing / unreadable / unparseable" with ``None``; the caller
  owns everything after the parse.
* ``ProvenanceStore._load``. Its corrupt-file posture is quarantine-aside
  (D13: the file holds operator-declared origins that exist nowhere else,
  so healing to empty would destroy them), which requires distinguishing
  OSError from JSONDecodeError — it keeps its own read path and delegates
  only its save here.
* Locking. Per-store ``threading.Lock`` discipline (T9b) stays with each
  store. The fixed ``.tmp`` sibling path means two unserialized writers
  to the SAME path race it and the loser's ``os.replace`` raises
  FileNotFoundError — ``write_atomic`` must be called under whatever
  serializes that store's writes.
"""

import json
import os
from pathlib import Path


def load_json(path: Path):
    """The file's parsed JSON, or None when it is missing, unreadable, or
    not JSON. A file containing literal ``null`` also reads as None — every
    adopter's shape heal already collapsed that case to its empty shape."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_atomic(path: Path, text: str, *, chmod: int | None = None) -> None:
    """mkdir → write a ``.tmp`` sibling → ``os.replace`` into place.

    ``chmod`` is applied to the tmp file BEFORE the replace, so the final
    path never exists with looser permissions, not even for one rename's
    duration (node_store's credential sidecar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    if chmod is not None:
        os.chmod(tmp, chmod)
    os.replace(tmp, path)


def save_json(path: Path, data, *, indent: int | None = None,
              sort_keys: bool = False, chmod: int | None = None) -> None:
    write_atomic(path, json.dumps(data, indent=indent, sort_keys=sort_keys),
                 chmod=chmod)
