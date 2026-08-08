"""Weights provenance — GGUFs, HF checkpoint directories, ComfyUI assets.

The honest asymmetry (design decision D5): size+mtime is a FINGERPRINT, not
a version. Calling it one would be the same category error as calling
docker's content id a version string. So ``version`` here is the sha256 or
``None``, the fingerprint lives in ``detail``, and a routine pass grades
CONSISTENT — "the cheap check passed, this is not proof". Only the
on-demand deep check (POST /api/provenance/verify, which hashes the file
with app.mover.hash_file) can produce EXACT.

``source_kind: "unknown"`` is first-class and is the default. Nothing here
may ever infer that a GGUF "probably came from" some HF repo (D8).
"""

from app import origins

SOURCE_KINDS = ("hf", "url", "unknown")


def origin(source_kind: str = "unknown", repo_id: str | None = None,
           revision: str | None = None, filename: str | None = None,
           url: str | None = None) -> dict:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(
            f"source_kind must be one of {list(SOURCE_KINDS)}, got {source_kind!r}")
    return {"source_kind": source_kind, "repo_id": repo_id,
            "revision": revision, "filename": filename, "url": url}


def identity_from_unit(unit: dict) -> dict:
    """A catalog unit (app.catalog._build_units) -> identity. No new I/O:
    the catalog scan already stat()'d this file."""
    return {"version": None,
            "label": unit.get("name"),
            "detail": {"size_bytes": unit.get("size"), "mtime": unit.get("mtime")}}


def grade(identity: dict, *, available: bool) -> str:
    detail = identity.get("detail") or {}
    if detail.get("size_bytes") is None:
        return origins.UNKNOWN
    return origins.CONSISTENT if available else origins.UNAVAILABLE


def matches_recorded(observed_sha: str, recorded_sha: str | None) -> bool | None:
    """Whether an on-demand hash agrees with what was already recorded.
    ``None`` when nothing was recorded — a first deep check has nothing to
    disagree with.

    Deliberately NOT a verification state. The state after a deep check is
    always EXACT either way: the file WAS hashed directly, so the new sha
    describes the current bytes exactly. ``False`` here says the BYTES
    CHANGED since the last record — a fact about the file, not a doubt about
    the check — and the caller records the new sha. An earlier draft returned
    CONSISTENT on a mismatch, which read as "we are less sure now" when the
    opposite is true; the store recorded EXACT regardless, so the two
    disagreed about the same event."""
    if recorded_sha is None:
        return None
    return observed_sha == recorded_sha
