"""Nodes router — the registry's CRUD + the test-connection probe + (E1
Task 10) the local node's declared-engines CRUD and the engine-kinds
catalog.

Same conventions as the sibling routers: deps from request.app.state.deck,
GuardError/ValueError left to the app-wide handlers (409/422), no auth.

THE CREDENTIAL IS WRITE-ONLY THROUGH THIS API. It is accepted on POST/PUT,
surfaced only as `credential_set`, and never appears in any response, error
body, or event detail — tests/test_nodes_router.py greps the literal out of
every path (a 422 for a missing required field also cannot echo it: see the
RequestValidationError handler in app/main.py, which strips pydantic's
`input` key from every error for exactly this reason). For the credential
specifically: `node-updated`'s event detail carries the field NAME
("credential") when one was supplied, never its value — other fields
(e.g. `label`) may still appear by value elsewhere (node-added logs
`label`), so this guarantee is about the credential, not every field.

Event kinds ride ui/src/model/eventSeverity.ts's suffix convention
(-failed => failure severity), so the Events tab needs zero new mapping.

E1 Task 10 — declared-engines CRUD (spec §1/§5/§6.2): the engine bodies
below (`add_engine`/`update_engine`) are bound as a bare `dict`, never a
strict pydantic model, so a shape defect is diagnosed by
`app.engine_kinds.validate_engines` — the single module allowed to know a
kind's connection schema (spec §8) — and its ONE-LINE message surfaces
verbatim. A stricter pydantic body model would validate (and word its
errors) before validate_engines ever ran. The resulting ValueError is
re-raised as a `RequestValidationError` (never left to the app-wide bare
ValueError handler) so it goes through app.main's REDACTING handler —
the same posture app.routers.sets.create_set's hand-rolled
`ConfigSet.model_validate` established (that module's docstring): today's
connection fields (url, metrics_url, container) carry no secret, but the
route makes no kind-specific exception, since E2 opens this schema up to
kinds that might.

`forget` (DELETE) is bookkeeping-only (spec §6.2 / coexistence ruling): it
drops the declaration entry plus the deck's OWN bookkeeping for it — the
intent record and the stored policy row — and nothing else (settings
scopes/provenance/events all survive, same posture as NodeStore.remove's
own docstring). It never calls the engine — no client lookup appears
anywhere in `forget_engine` below.

`kinds_router` is a SEPARATE, unprefixed router (mounted at `/api` in
app.main alongside `router` below) so `GET /engine-kinds` lands at
`/api/engine-kinds`, not nested under `/api/nodes` — it is the UI's kind
picker source (spec §5), not a per-node resource.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from app.engine_kinds import ENGINE_KINDS, KNOWN_KINDS, validate_engines
from app.engines import EngineError
from app.events import log_event
from app.observe import local_key

router = APIRouter(prefix="/nodes", tags=["nodes"])
kinds_router = APIRouter(tags=["engine-kinds"])


class NodeCreate(BaseModel):
    id: str
    label: str
    address: str
    serving_address: str | None = None
    credential: str | None = None
    # Declared operability (app/node_store.py _CONTROLS). Default "none":
    # adding a node never grants verbs implicitly.
    control: str = "none"


class NodePatch(BaseModel):
    label: str | None = None
    address: str | None = None
    serving_address: str | None = None
    credential: str | None = None
    control: str | None = None


class NodeTestBody(BaseModel):
    node_id: str | None = None
    address: str | None = None
    credential: str | None = None


def _public(store, entry: dict) -> dict:
    return {**entry, "credential_set": store.credential_set(entry["id"])}


def _checked_credential(value: str | None) -> str | None:
    """None = not provided. "" = provided and wrong: refused, never coerced
    to 'absent' (an empty credential is not a credential)."""
    if value is None:
        return None
    if not value:
        raise ValueError("credential must be a non-empty string when provided")
    return value


@router.get("")
def list_nodes(request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["node_store"]
    return {"nodes": [_public(store, n) for n in store.list()]}


@router.post("")
def create_node(body: NodeCreate, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["node_store"]
    credential = _checked_credential(body.credential)
    entry = store.add({"id": body.id, "label": body.label,
                       "agent_kind": "node-agent", "address": body.address,
                       "serving_address": body.serving_address,
                       "control": body.control},
                      credential=credential)
    log_event(deck["events_path"], "node-added",
              {"node": entry["id"], "label": entry["label"]})
    return _public(store, entry)


@router.put("/{node_id}")
def update_node(node_id: str, body: NodePatch, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["node_store"]
    fields = body.model_dump(exclude_unset=True)
    credential = _checked_credential(fields.pop("credential", None))
    entry = store.update(node_id, fields, credential=credential)
    changed = sorted(fields) + (["credential"] if credential else [])
    log_event(deck["events_path"], "node-updated",
              {"node": node_id, "fields": changed})   # names, never values
    return _public(store, entry)


@router.delete("/{node_id}")
def delete_node(node_id: str, request: Request) -> dict:
    deck = request.app.state.deck
    # No residual binding to surface: app.node_clients rebinds every
    # actuation client live, from the registry, on every call — a deleted
    # row simply stops resolving to a client on client_for's NEXT call, no
    # restart involved. The store's own guard (NodeStore.remove) already
    # refuses deleting an operable (control:"swap") node outright, so a row
    # reaching this point has nothing for a client to remain bound to.
    deck["node_store"].remove(node_id)
    log_event(deck["events_path"], "node-removed", {"node": node_id})
    return {"status": "ok"}


@router.post("/test")
def test_connection(body: NodeTestBody, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["node_store"]
    if body.node_id and body.address:
        # Literal-and-declared: two targets in one call is ambiguous input,
        # not "node_id wins" — refuse rather than silently pick one.
        raise ValueError("provide node_id or address, not both")
    if body.node_id:
        entry = store.get(body.node_id)
        if entry is None:
            raise ValueError(f"unknown node {body.node_id!r}")
        address = entry.get("address") or ""
        key = store.credential_for(body.node_id)
        target = body.node_id
    elif body.address:
        # Never-coerce: an explicitly-empty credential is already refused
        # below by _checked_credential ("credential must be a non-empty
        # string when provided"). A credential that's simply ABSENT used to
        # fall through as `None or ""` — an empty bearer the node-agent
        # 401s on, surfacing as an unhelpful {"ok": false, "error": ""}.
        # Refuse that too, with the same message the "neither target"
        # branch below uses: both are "an address with no way to
        # authenticate it".
        key = _checked_credential(body.credential)
        if key is None:
            raise ValueError(
                "provide node_id (stored credential) or address+credential")
        address = body.address
        target = body.address
    else:
        raise ValueError("provide node_id (stored credential) or address+credential")
    if not address:
        raise ValueError("node has no address to test")

    client = deck["node_agent_client_factory"](address, key)
    try:
        info = client.info()
    except EngineError as exc:
        log_event(deck["events_path"], "node-test-failed",
                  {"target": target, "note": type(exc).__name__})
        return {"ok": False, "error": str(exc)}
    finally:
        client.close()
    return {"ok": True, "name": info.get("name"), "platform": info.get("platform"),
            "capabilities": info.get("capabilities", []),
            "gpu_count": len(info.get("gpus") or [])}


# ===========================================================================
# E1 Task 10 — declared-engines CRUD (local node only, spec §1/§4)
# ===========================================================================


def _local_engines(deck: dict) -> list[dict]:
    """The local node's `engines[]` declaration, read LIVE off node_store —
    never a boot-time copy (same posture as app.routers.control._declared_kind
    / app.routers.build_world_snapshot)."""
    local = deck["node_store"].get("local")
    return list(local.get("engines", [])) if local is not None else []


def _shape_error(msg: str, *loc: str) -> RequestValidationError:
    """Wrap a body-shape defect as a RequestValidationError (module
    docstring: never the bare ValueError handler for a request body) —
    `loc` mirrors pydantic's own error-location tuple convention."""
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body", *loc), "msg": msg}])


@router.post("/local/engines")
def add_engine(body: dict, request: Request) -> dict:
    """Declare a new local engine. 422 (validate_engines' own message) for
    a shape defect; 409 for a resource already declared — checked AFTER
    shape validation (NodeStore.add's own order: full validate, then
    duplicate-identity check)."""
    deck = request.app.state.deck
    store = deck["node_store"]
    try:
        validate_engines([body])
    except ValueError as exc:
        raise _shape_error(str(exc)) from exc
    resource = body["resource"]
    engines = _local_engines(deck)
    if any(e["resource"] == resource for e in engines):
        raise HTTPException(status_code=409,
                            detail=f"engine {resource!r} already exists")
    entry = store.update("local", {"engines": engines + [body]})
    log_event(deck["events_path"], "engine-added",
              {"resource": resource, "kind": body["kind"]})
    return next(e for e in entry["engines"] if e["resource"] == resource)


@router.put("/local/engines/{resource}")
def update_engine(resource: str, body: dict, request: Request) -> dict:
    """Full-entry replace. `resource` in the body must equal the path — a
    declared resource's identity is immutable (it keys `local/<resource>`
    intent, policy, and settings scopes, the same rationale
    app/node_store.py's `id` docstring states for nodes) — a mismatch is
    refused as a rename attempt (422), never coerced to either side
    ([[literal-declared-inputs]]). 404 when the path resource isn't
    currently declared.

    E1 final-review item 5 (controller ruling): a KIND change invalidates
    the OLD kind's intent record. app/state.py's observation half already
    self-heals a kind mismatch on read (`_KIND_MEM_KEY`, World.snapshot's
    per-tick comparison); the intent (goal) half has no equivalent lazy
    check — `app.reconcile.plan_reconcile` copies `intent["engine"]`
    straight into the restore action, and `app.arbiter` (:1178) resolves
    the adapter off THAT name, not the live declaration — so a stale
    old-kind record surviving a kind change can drive `_restore` through
    the WRONG adapter for this resource. Contained today by the failure
    budget/quarantine (noisy, not silent), but avoidable outright: forget
    the intent record here — the exact call `forget_engine` below uses
    (`IntentStore.forget`) — whenever the incoming `kind` differs from what
    is currently declared. A same-kind edit (moving `gpu_index`,
    `policy_defaults`, connection fields) must NOT touch it — only an
    actual kind change invalidates the record. See the crash-ordering
    comment at the forget call below for why it runs BEFORE the
    declaration write, not after (the opposite order from `forget_engine`'s
    own three-write sequence, and deliberately so)."""
    deck = request.app.state.deck
    store = deck["node_store"]
    if body.get("resource") != resource:
        raise _shape_error(
            f"resource {body.get('resource')!r} must match the path "
            f"{resource!r} — rename is refused; forget and re-add instead",
            "resource")
    engines = _local_engines(deck)
    existing = next((e for e in engines if e["resource"] == resource), None)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"unknown engine {resource!r}")
    try:
        validate_engines([body])
    except ValueError as exc:
        raise _shape_error(str(exc)) from exc
    if body["kind"] != existing["kind"]:
        # Forget BEFORE the declaration write below, not after — reversed
        # from forget_engine's own ordering (declaration first, THEN
        # intent), because the two routes' crash windows have asymmetric
        # worst cases, not the same one. Declaration-first HERE would mean:
        # a crash between the two writes leaves this resource declared
        # under the NEW kind while intent still carries the OLD kind's
        # `engine` field — exactly the live bug this fix exists to close,
        # reintroduced for the length of the crash window (a reconciler
        # tick landing there sees "down" under the new kind's observation
        # and restores through the old adapter). Intent-first means: a
        # crash here leaves the OLD kind still declared (the write below
        # never ran, so nothing actually changed) with no intent record for
        # it — inert, "the deck has no opinion", the exact safe-by-default
        # reading `app.reconcile.plan_reconcile` already gives an ABSENT
        # intent (skip, never invent one — see that module's own comment).
        # Worst case with this order is silence (no auto-restore until the
        # operator retries the edit or acts on the resource again), never a
        # wrong-adapter actuation — strictly safer than the reversed order,
        # same standard forget_engine's own analysis applies, just landing
        # on the opposite conclusion because what is at risk differs (there:
        # an orphaned policy override; here: an actively wrong actuation).
        deck["intent_store"].forget(local_key(resource))
    new_engines = [body if e["resource"] == resource else e for e in engines]
    entry = store.update("local", {"engines": new_engines})
    log_event(deck["events_path"], "engine-updated", {"resource": resource})
    return next(e for e in entry["engines"] if e["resource"] == resource)


@router.delete("/local/engines/{resource}")
def forget_engine(resource: str, request: Request) -> dict:
    """Forget (spec §6.2 / coexistence ruling — module docstring):
    bookkeeping only. Drops the declaration entry, the intent record, and
    the stored policy row — nothing else. Never calls the engine: no
    client lookup anywhere in this function. 404 when unknown."""
    deck = request.app.state.deck
    store = deck["node_store"]
    engines = _local_engines(deck)
    if not any(e["resource"] == resource for e in engines):
        raise HTTPException(status_code=404, detail=f"unknown engine {resource!r}")
    new_engines = [e for e in engines if e["resource"] != resource]
    # Declaration first, THEN intent, THEN policy — crash safety across
    # these three separate writes, NOT a same-call race: PolicyStore.forget
    # is correct regardless of this ordering by itself (its own docstring)
    # — its pop runs on the same dict its own heal read, inside the same
    # lock, so there is nothing here for a still-declared resource to
    # "re-materialize" between read and pop within ONE call.
    #
    # What the ordering actually decides is which PARTIAL state a crash
    # BETWEEN these three writes leaves behind. Reversed (policy/intent
    # forgotten first, declaration removed last): a crash after the policy
    # row is popped but before the declaration write leaves a
    # STILL-DECLARED — possibly still loaded — resource whose stored
    # override is gone; PolicyStore's declared-defaults overlay
    # re-materializes it at DEFAULTS on the very next read, so a pinned
    # resource silently becomes evictable out from under an operator who
    # never asked to un-pin it. This order's worst case is strictly safer:
    # an orphaned, invisible policy row for a resource already gone from
    # the deck's view (PolicyStore's own documented "undeclared row
    # survives, invisible" posture) — inert, not a safety regression.
    store.update("local", {"engines": new_engines})
    # Orphaned-intent window (accepted, not closed): IntentStore keeps no
    # declared-resource gate of its own, so a crash right here — after the
    # declaration write above, before the forget below runs — leaves an
    # intent record for a name nothing declares anymore. Inert today
    # because the reconciler only ever joins intent against OBSERVED
    # state, and observation is itself declaration-driven (app.observe) —
    # an undeclared name is observed nowhere. The one way it stops being
    # inert: the SAME resource name gets re-declared later (possibly under
    # a different kind), and the stale record surfaces against the new
    # resource's observations. Accepted: it takes both a crash at this
    # exact point AND that name reuse, the redeclare is the operator's own
    # deliberate act, and derive_status/settings_drift are report-only —
    # nothing actuates off a stale intent on its own.
    deck["intent_store"].forget(local_key(resource))
    deck["policy_store"].forget(resource)
    log_event(deck["events_path"], "engine-removed", {"resource": resource})
    return {"status": "ok"}


@kinds_router.get("/engine-kinds")
def list_engine_kinds() -> dict:
    """The UI's kind picker source (spec §5): every known kind's connection
    schema (field -> required) and its human-initiated verb vocabulary,
    served from app.engine_kinds — the one module allowed to know engine
    names (spec §8) — so the picker never bakes a kind name into the UI."""
    kinds = []
    for kind in sorted(KNOWN_KINDS):
        schema = KNOWN_KINDS[kind]
        kinds.append({
            "kind": kind,
            "connection": {field: {"required": required}
                          for field, required in schema.items()},
            "human_verbs": sorted(ENGINE_KINDS[kind].human_verbs()),
        })
    return {"kinds": kinds}
