"""Update-checking vocabulary.

FOUR STATUS WORDS, AND TWO OF THEM ARE NOT SYNONYMS. `undetermined` means the
upstream was reached and its answer cannot be RANKED (an `order: none` tag
set); `unavailable` means it could not be reached or could not be asked. This
is the same distinction app.origins already draws between UNKNOWN and
UNAVAILABLE, and collapsing them would hide which one an operator can fix.

Provenance NEVER actuates, and neither does this: the package reads upstreams
and writes a record. app.reconcile remains the only actuator.
"""

from app.updates.ordering import ORDERS

CURRENT = "current"              # checked; nothing newer
AVAILABLE = "available"          # checked; something newer exists
UNDETERMINED = "undetermined"    # reached it; cannot rank what it found
UNAVAILABLE = "unavailable"      # could not reach or could not ask

STATUSES = (CURRENT, AVAILABLE, UNDETERMINED, UNAVAILABLE)

CHECKS = ("oci_channel", "oci_tags", "git_compare", "git_tags")
_TAG_CHECKS = ("oci_tags", "git_tags")

# WHAT EACH CHECKER ACTUALLY READS -- derived by reading the checkers, never
# from the docs, because a table that disagrees with them is worse than no
# table. `registry` is deliberately absent: both oci checkers default it
# (app/updates/oci.py:59,86 -- `source.get("registry") or "ghcr.io"`), and a
# field the checker defaults is not a field the operator must supply.
#
#   oci_channel  app/updates/oci.py:60-62   repository / reference / pinned
#   oci_tags     app/updates/oci.py:87-88   repository / pinned  (+ order, below)
#   git_compare  app/updates/git.py:101,105,107  remote / pinned / ref
#   git_tags     app/updates/git.py:135,139   remote / pinned      (+ order, below)
#
# `remote` is `.get("remote", "")` rather than an index, so it never raises --
# it produces a PERMANENT "remote is not on github.com; no checker available"
# instead (app/updates/git.py:44-46,102-103). Same category: a source no checker
# can execute, refused at the door rather than written to disk.
_REQUIRED = {
    "oci_channel": ("repository", "reference", "pinned"),
    "oci_tags": ("repository", "pinned"),
    "git_compare": ("remote", "ref", "pinned"),
    "git_tags": ("remote", "pinned"),
}

# Worst-first. A rollup is the worst of its sources so that one healthy input
# cannot mask a sibling -- lemonade has two that drift independently.
_SEVERITY = {UNAVAILABLE: 3, UNDETERMINED: 2, AVAILABLE: 1, CURRENT: 0}


class BadWatch(ValueError):
    """A watch source that cannot be checked as written."""


def validate_watch_sources(sources: list[dict]) -> None:
    """Raise BadWatch unless `sources` is a checkable REPLACEMENT watch list.

    UNIQUENESS IS A PROPERTY OF THE LIST, so it cannot live in
    `validate_watch`, which only ever sees one source. `record_update` merges
    a pass's results into `{s["id"]: s}` (app/provenance.py:511) and
    `set_watch` narrows the list to a set of ids (app/provenance.py:466), so
    two sources sharing an id means one of them silently has no verdict --
    the README published `id` as "unique within the artifact" and nothing
    enforced it. Refused, not deduped: which of the two the operator meant is
    exactly the ambiguity this project does not guess at.
    """
    seen: set[str] = set()
    for source in sources:
        validate_watch(source)
        source_id = source["id"]         # a str by now: validate_watch said so
        if source_id in seen:
            raise BadWatch(f"duplicate watch source id {source_id!r}")
        seen.add(source_id)


def validate_watch(source: dict) -> None:
    """Raise BadWatch unless `source` is a checkable watch entry."""
    source_id = source.get("id")
    if not isinstance(source_id, str) or not source_id:
        # Truthiness alone (the original check) let a list or dict `id`
        # through -- valid JSON, invalid as a key. ProvenanceStore.set_watch
        # builds `{s["id"] for s in sources}` right after this validates, and
        # an unhashable id there is an unhandled TypeError (500), not the
        # clean 422 every other malformed watch body gets. Task 9 owns the
        # route that surfaces it, so the fix belongs here at the one
        # validator every write path already calls.
        raise BadWatch("watch source needs an id")
    check = source.get("check")
    if check not in CHECKS:
        raise BadWatch(f"check must be one of {list(CHECKS)}, got {check!r}")

    # A source whose checker cannot execute it is refused HERE, not accepted
    # with a 200 and turned into a permanent `unavailable` carrying "checker
    # raised KeyError" -- dispatch's per-source try (app/update_check.py:64-66)
    # swallows the KeyError, so nothing downstream can ever tell an operator
    # their body was wrong. Non-string is refused for the same reason a
    # non-string `id` is: `ordering.rank` regexes the pin and `parse_remote`
    # strips the remote, both of which raise on anything else.
    for field in _REQUIRED[check]:
        value = source.get(field)
        if not isinstance(value, str) or not value:
            raise BadWatch(
                f"{check} source {source_id!r} needs a non-empty string "
                f"{field!r}, got {value!r}")

    order = source.get("order")
    if check in _TAG_CHECKS:
        if order not in ORDERS:
            raise BadWatch(
                f"{check} needs order one of {list(ORDERS)}, got {order!r}")
    elif order is not None:
        raise BadWatch(f"{check} takes no order, got {order!r}")


def rollup(source_statuses: list[str]) -> str:
    """The worst status among `source_statuses`; UNAVAILABLE when empty --
    an artifact with nothing to watch was never checked, which is not the
    same as being current.

    Always returns a member of STATUSES. A value that is not one of the four
    (e.g. corrupt or hand-edited provenance.json) normalizes to UNAVAILABLE
    rather than being raised or returned verbatim: this is called from the
    update-check pass over stored data, and nothing about update-checking may
    fail a tick, block a swap, or touch intent. An unreadable status
    degrading to "we do not know" is both safe and honest -- treating it as
    the worst case is exactly what UNAVAILABLE already means. Normalizing
    before ranking (rather than defaulting only the missing severity) also
    keeps the result order-independent: an unrecognized value and a literal
    "unavailable" now compare equal, so tied inputs no longer depend on which
    one happened to come first in the list.
    """
    if not source_statuses:
        return UNAVAILABLE
    normalized = (s if s in STATUSES else UNAVAILABLE for s in source_statuses)
    return max(normalized, key=lambda s: _SEVERITY[s])
