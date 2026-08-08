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


def grade_deep(observed_sha: str, recorded_sha: str | None) -> str:
    """The on-demand check. A first deep check ESTABLISHES the sha (there
    was nothing to disagree with); a mismatch drops back to CONSISTENT —
    the bytes changed, so the recorded sha describes a different file and
    the caller records the new one."""
    if recorded_sha is None or observed_sha == recorded_sha:
        return origins.EXACT
    return origins.CONSISTENT
