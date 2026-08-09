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

# Worst-first. A rollup is the worst of its sources so that one healthy input
# cannot mask a sibling -- lemonade has two that drift independently.
_SEVERITY = {UNAVAILABLE: 3, UNDETERMINED: 2, AVAILABLE: 1, CURRENT: 0}


class BadWatch(ValueError):
    """A watch source that cannot be checked as written."""


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
    if not source.get("pinned"):
        raise BadWatch(f"watch source {source['id']!r} needs a pinned value")

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
