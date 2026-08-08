"""Origin kinds — the three ways an artifact on this box can have come from
somewhere else, and the vocabulary shared across them.

Four artifact classes were in scope (weights, engine images, source repos,
ComfyUI node packs) and they collapse to THREE kinds: a node pack IS a git
checkout, so pinning KJNodes is a git-ref problem, not a fourth mechanism
(design decision D7). ``role`` keeps the four-way class distinction as data
for recipes, without a fourth code path.

Every module here is PURE — values in, values out, no I/O, no clients, no
clock. That is deliberate: it is what lets the whole layer be tested from a
table, and it keeps the one place that reaches the network (the collector)
separate from the one place that decides what a reading means.

``artifact_id`` is ``<kind>:<node>:<ref>`` and splits on the FIRST TWO
colons only — a weights relpath legitimately contains ``/`` and can contain
``:``. The ref is deliberately NOT the storage location: ``app.catalog``'s
unit ids embed the location name and ``record_moved()`` rewrites them on a
move (app/catalog.py:175-185), so keying provenance on a unit id would
orphan a model's origin exactly when the any-drive mover runs. ``relpath``
survives a move; the location is a current-placement fact, not identity.
"""

import re

KINDS = ("oci", "git", "file")
ROLES = ("engine", "weights", "source", "nodepack", "other")

# Verification states. The first four are stored; STALE is computed at read
# time from ``verified_at`` (app.provenance.describe), the same way
# app.locations.describe() computes ``available`` rather than persisting it.
EXACT = "exact"              # machine identity read directly this pass
CONSISTENT = "consistent"    # cheap check passed; not proof (weights, D5)
UNAVAILABLE = "unavailable"  # source unreachable — NOT the same as unknown
UNKNOWN = "unknown"          # never observed
STALE = "stale"              # derived at read time

_NODE_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")   # app.locations' slug rule


class BadArtifactId(ValueError):
    """A malformed artifact id. Refused, never coerced into a guess (D8)."""


def parse_artifact_id(artifact_id: str) -> tuple[str, str, str]:
    """``"oci:sparky:ds4-spark"`` -> ``("oci", "sparky", "ds4-spark")``."""
    parts = (artifact_id or "").split(":", 2)
    if len(parts) != 3:
        raise BadArtifactId(
            f"artifact id must be '<kind>:<node>:<ref>', got {artifact_id!r}")
    kind, node, ref = parts
    if kind not in KINDS:
        raise BadArtifactId(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    if not _NODE_RE.match(node):
        raise BadArtifactId(f"node {node!r} must be a lowercase slug ([a-z0-9-])")
    if not ref:
        raise BadArtifactId("artifact ref must not be empty")
    return kind, node, ref


def build_artifact_id(kind: str, node: str, ref: str) -> str:
    """Inverse of parse_artifact_id, validated the same way."""
    artifact_id = f"{kind}:{node}:{ref}"
    parse_artifact_id(artifact_id)
    return artifact_id
