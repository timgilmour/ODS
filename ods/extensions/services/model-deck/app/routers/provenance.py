"""Provenance router — read the ledger, declare what the deck cannot derive,
and ask for a deep check.

``artifact_id`` is NEVER a URL path segment. It contains ``:`` by
construction and legitimately contains ``/`` (a weights relpath) and ``~``
(a repo path), so it travels as a query parameter on reads and deletes and
as a body field on writes. That is a deliberate shape choice, not an
oversight: percent-encoding it into a path segment invites exactly the class
of routing bug that is expensive to find later.

Nothing here actuates. ``PUT /desired`` records what an artifact SHOULD be
at; no code converges to it — see app.arbiter.Watcher._provenance_pass for
why that is a permissions question and not a missing feature.

``BadArtifactId`` subclasses ValueError, which app.main already maps to 422
app-wide, so a malformed id is refused rather than coerced (the standing
rule that ambiguous input is refused, never guessed at).
"""

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import origins, provenance, provenance_history
from app.mover import hash_file
from app.origins import file as file_origin

router = APIRouter(prefix="/provenance", tags=["provenance"])


class OriginBody(BaseModel):
    artifact_id: str
    role: str
    origin: dict | None
    update_path: str | None = None
    notes: str | None = None


class DesiredBody(BaseModel):
    artifact_id: str
    version: str | None
    label: str | None = None


class ArtifactBody(BaseModel):
    artifact_id: str


def _store(request: Request):
    return request.app.state.deck["provenance_store"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@router.get("")
def list_provenance(request: Request) -> dict:
    deck = request.app.state.deck
    data = deck["provenance_store"].get()
    return {
        "artifacts": provenance.describe(
            data, now=_now(), stale_s=deck["settings"].provenance_stale_s),
        "gaps": provenance.gaps(data),
    }


@router.get("/history")
def get_history(request: Request, artifact_id: str) -> dict:
    """Every recorded transition for one artifact, oldest first — the
    restorable record behind "what was this running last Tuesday"."""
    deck = request.app.state.deck
    records = provenance_history.history_for(
        Path(deck["provenance_history_path"]), artifact_id)
    return {"artifact_id": artifact_id, "records": records}


@router.post("/backfill")
def backfill(request: Request) -> dict:
    """Create an entry for every artifact the deck can already see, with
    ``origin: null``.

    It never fills in an origin — that is the point. The output IS the gap
    list: "41 artifacts, 39 with unknown origin" is the honest state of a box
    that has been accreting artifacts for a year with no record, and that
    number going down is the operator's progress bar.

    Idempotent: re-running observes the same identities, and
    ProvenanceStore.observe writes history only on a real change, so a second
    run adds no records.

    Deliberately does NOT reach sparky. A backfill is an operator action
    expected to return promptly, and the sparky sweep is N+2 network
    round-trips against a box that may be mid-swap; the watcher pass covers
    it within provenance_interval_s anyway.
    """
    from app import provenance_collect

    deck = request.app.state.deck
    store = deck["provenance_store"]
    before = set(store.get())

    node = deck["settings"].node_label
    entries = list(provenance_collect.local_file_entries(
        deck["catalog"].units(), node))

    dockerctl = deck.get("dockerctl")
    if dockerctl is not None:
        bodies: dict[str, dict | None] = {}
        for name in deck["settings"].park_allowlist:
            try:
                bodies[name] = dockerctl.inspect(name)
            except Exception:  # noqa: BLE001 — a container we cannot read is skipped
                bodies[name] = None
        entries += provenance_collect.local_oci_entries(bodies, node)

    for entry in entries:
        current = entry.pop("current")
        store.observe(**entry, current=current)

    data = store.get()
    return {"created": len(set(data) - before), "total": len(data),
            "gaps": len(provenance.gaps(data))}


@router.put("/origin")
def put_origin(request: Request, body: OriginBody) -> dict:
    """An operator asserts where an artifact came from. This is how the
    hand-maintained registry in sparky-vllm/README.md — including
    aeon-vllm's "no source repo available" and its cold-storage archive —
    stops being a file somebody has to remember to edit."""
    kind, node, _ = origins.parse_artifact_id(body.artifact_id)
    return _store(request).declare_origin(
        body.artifact_id, kind=kind, node=node, role=body.role,
        origin=body.origin, update_path=body.update_path, notes=body.notes)


@router.put("/desired")
def put_desired(request: Request, body: DesiredBody) -> dict:
    origins.parse_artifact_id(body.artifact_id)
    try:
        return _store(request).set_desired(
            body.artifact_id, version=body.version, label=body.label)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"unknown artifact {body.artifact_id!r}") from None


@router.delete("/desired")
def delete_desired(request: Request, artifact_id: str) -> dict:
    """'No opinion' — app.intent.forget's semantics for a desired version."""
    _store(request).clear_desired(artifact_id)
    return {"artifact_id": artifact_id, "desired": None}


@router.post("/verify")
def deep_verify(request: Request, body: ArtifactBody) -> dict:
    """Hash the file and record the result — the ONLY path that can grade a
    weights artifact ``exact``. File kind only: there is nothing to hash for
    an image or a repo, and pretending otherwise would invent a verification
    that never happened."""
    kind, _node, relpath = origins.parse_artifact_id(body.artifact_id)
    if kind != "file":
        raise HTTPException(
            status_code=422,
            detail=f"deep verify applies to file artifacts, not {kind!r}")

    deck = request.app.state.deck
    store = deck["provenance_store"]
    entry = store.entry(body.artifact_id)
    if entry is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown artifact {body.artifact_id!r}")

    path = _resolve_unit_path(deck, relpath)
    if path is None:
        raise HTTPException(
            status_code=409,
            detail="no available location currently holds this file")

    observed = hash_file(path)
    matched = file_origin.matches_recorded(
        observed, (entry["current"] or {}).get("version"))
    updated = store.record_deep_verify(body.artifact_id, observed)
    return {"artifact_id": body.artifact_id, "sha256": observed,
            # None = nothing was recorded to compare against; False = the
            # bytes changed since the last deep check. Either way the entry
            # is now EXACT — the file was just hashed.
            "matched_recorded": matched, "entry": updated}


@router.delete("")
def delete_artifact(request: Request, artifact_id: str) -> dict:
    """The only way an entry leaves the ledger. History is never deleted."""
    _store(request).delete(artifact_id)
    return {"artifact_id": artifact_id, "deleted": True}


def _resolve_unit_path(deck: dict, relpath: str) -> Path | None:
    """Current placement of a weights unit, or None if no AVAILABLE location
    holds it. Looked up fresh rather than stored on the entry — location is a
    placement fact and the mover changes it, which is the whole reason
    provenance keys on relpath (app.provenance_collect.local_file_entries)."""
    location_store = deck["location_store"]
    locations = {loc["name"]: loc for loc in location_store.list()}
    for unit in deck["catalog"].units():
        if unit.get("relpath") != relpath:
            continue
        loc = locations.get(unit.get("location"))
        if loc and location_store.available(loc):
            return Path(loc["path"]) / relpath
    return None
