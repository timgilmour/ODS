"""Deck-created engine INSTANCES (INST I1, design §4-§5): create / remove /
move. Declaration + actuation in one place; the declaration IS the
instance record (a managed engines[] entry), the actuation is one POST to
the node whose registry entry declares control:"instances" — looked up
through deck["node_clients"] exactly like a swap (no "local" literal: the
node id is whatever the path says).

Ordering (D-I1-1):
  create: declare -> ship; agent refusal -> roll the declaration back (502).
  remove: hold (reconciler must not restore what we are tearing down) ->
          ship -> forget declaration/intent/policy (forget_engine's sequence).
  move:   ship the NEW claim -> update the declaration -> forget intent (a
          moved container is a NEW container; the operator reloads).
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.engine_kinds import (KNOWN_KINDS, gpu_indices_of, instance_connection, instance_policy,
                              kind_instantiable, validate_engines)
from app.engines import BusyError, EngineError, GuardError
from app.events import log_event
from app.instances import allocate_port, check_observed_gpus, instance_document, next_resource_name
from app.node_clients import control_prereqs
from app.observe import node_key
from app.routers.nodes import _node_or_404, _observed_gpu_indices, _reobserve, _shape_error

router = APIRouter(prefix="/nodes/{node_id}/instances", tags=["instances"])
_HOLD_S = 120.0


class InstanceCreate(BaseModel):
    kind: str
    gpu_indices: list[int] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)


class InstanceMove(BaseModel):
    gpu_indices: list[int] = Field(min_length=1)


def _client(deck: dict, node: dict):
    if node.get("control") != "instances":
        raise HTTPException(status_code=503, detail=(
            f'node {node["id"]!r} is not operable for instances: needs control: "instances" '
            f'with {", ".join(control_prereqs("instances"))}, a credential and instance_port_range'))
    client = deck["node_clients"].client_for(node["id"])
    if client is None:
        raise HTTPException(status_code=503, detail=(
            f'node {node["id"]!r} is declared control: "instances" but a prerequisite is missing '
            '(address, credential)'))
    return client


def _managed_entry(node: dict, resource: str) -> dict:
    entry = next((e for e in node.get("engines", []) if e["resource"] == resource), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown engine {resource!r}")
    if not entry.get("managed"):
        raise HTTPException(status_code=409, detail=(
            f"{resource!r} is not a deck-managed instance — use forget (DELETE /engines/{resource}) "
            "for a declared-only engine"))
    return entry


def _ship(deck, client, verb, entry, fail_kind, node_id, extra: dict | None = None):
    """Ship one verb's document to the node's actuation client. `extra`
    (module docstring's event-kind table) carries the ONE field that
    differs between the three failure events — create's needs `kind`,
    remove's and move's don't — so this stays the single EngineError ->
    502 mapping point rather than three near-duplicate try/excepts."""
    try:
        client.request(verb, instance_document(entry))
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EngineError as exc:
        log_event(deck["events_path"], fail_kind,
                  {"node": node_id, "resource": entry["resource"], "error": str(exc), **(extra or {})})
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("", status_code=201)
def create_instance(node_id: str, body: InstanceCreate, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["node_store"]
    node = _node_or_404(deck, node_id)
    client = _client(deck, node)
    kind = body.kind
    if kind not in KNOWN_KINDS or not kind_instantiable(kind):
        raise _shape_error(f"kind {kind!r} is not instantiable (instantiable: "
                           f"{sorted(k for k in KNOWN_KINDS if kind_instantiable(k))})", "kind")
    taken = {e["resource"] for n in store.list() for e in n.get("engines", [])}
    resource = next_resource_name(kind, taken)
    engines = list(node.get("engines", []))
    try:
        port = allocate_port(node["instance_port_range"],
                             {e["port"] for e in engines if e.get("managed")})
    except GuardError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    entry = {"resource": resource, "kind": kind, "connection": instance_connection(kind, resource),
             "gpu_indices": sorted(body.gpu_indices), "policy_defaults": instance_policy(kind),
             "container_consent": True, "managed": True, "port": port, "env": dict(body.env)}
    try:
        validate_engines([entry], remote=node["agent_kind"] == "node-agent")
        check_observed_gpus(entry["gpu_indices"], _observed_gpu_indices(deck, node))
    except ValueError as exc:
        raise _shape_error(str(exc)) from exc
    store.update(node_id, {"engines": engines + [entry]})
    try:
        _ship(deck, client, "create", entry, "instance-create-failed", node_id, extra={"kind": kind})
    except HTTPException:
        store.update(node_id, {"engines": engines})      # roll back: nothing was launched
        raise
    _reobserve(deck, node)
    log_event(deck["events_path"], "instance-created",
              {"node": node_id, "resource": resource, "kind": kind,
               "gpu_indices": entry["gpu_indices"], "port": port})
    return entry


@router.delete("/{resource}")
def remove_instance(node_id: str, resource: str, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["node_store"]
    node = _node_or_404(deck, node_id)
    client = _client(deck, node)
    entry = _managed_entry(node, resource)
    key = node_key(node_id, resource)
    deck["hold_store"].hold(key, ttl_s=_HOLD_S)
    _ship(deck, client, "remove", entry, "instance-remove-failed", node_id)
    store.update(node_id, {"engines": [e for e in node.get("engines", []) if e["resource"] != resource]})
    deck["intent_store"].forget(key)
    deck["policy_store"].forget(resource)
    _reobserve(deck, node)
    log_event(deck["events_path"], "instance-removed", {"node": node_id, "resource": resource})
    return {"status": "ok"}


@router.post("/{resource}/move")
def move_instance(node_id: str, resource: str, body: InstanceMove, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["node_store"]
    node = _node_or_404(deck, node_id)
    client = _client(deck, node)
    entry = _managed_entry(node, resource)
    new_claim = sorted(body.gpu_indices)
    if new_claim == gpu_indices_of(entry):
        raise HTTPException(status_code=409, detail=f"{resource!r} already claims {new_claim}")
    moved = {**entry, "gpu_indices": new_claim}
    moved.pop("gpu_index", None)
    try:
        validate_engines([moved], remote=node["agent_kind"] == "node-agent")
        check_observed_gpus(new_claim, _observed_gpu_indices(deck, node))
    except ValueError as exc:
        raise _shape_error(str(exc), "gpu_indices") from exc
    key = node_key(node_id, resource)
    deck["hold_store"].hold(key, ttl_s=_HOLD_S)
    _ship(deck, client, "move", moved, "instance-move-failed", node_id)
    store.update(node_id, {"engines": [moved if e["resource"] == resource else e
                                       for e in node.get("engines", [])]})
    deck["intent_store"].forget(key)
    _reobserve(deck, node)
    log_event(deck["events_path"], "instance-move-requested",
              {"node": node_id, "resource": resource, "from": gpu_indices_of(entry), "to": new_claim})
    return moved
