"""Nodes router — the registry's CRUD + the test-connection probe + (E1
Task 10, node-scoped since E1 Task 5) the declared-engines CRUD and the
engine-kinds catalog.

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
# `node_key` imported rather than spelled here, the same way app.arbiter
# takes it: one definition of the `<node>/<resource>` key shape, next to the
# `local_key` builder that fixes its node half.
from app.observe import node_key

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
# E1 Task 10 — declared-engines CRUD (node-scoped since E1 Task 5, spec §1/§4)
# ===========================================================================


def _node_or_404(deck: dict, node_id: str) -> dict:
    """The registry entry for `node_id`, or a 404 (E1 Task 5: the
    declared-engines CRUD below is node-scoped now, not local-only, so an
    unknown node is a real "not found" here — same status the sibling
    unknown-engine checks already use for the resource half of this same
    URL space)."""
    node = deck["node_store"].get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
    return node


def _reobserve(deck: dict, node: dict) -> None:
    """Drop the cached REMOTE half of the world after a declaration edit on
    `node` (sglang-omni Task 8, deferred from Task 7's review).

    The remote half is TTL-cached 10 s (app.node_clients.RemoteObserver), so
    without this a newly declared engine is absent from the board — and a
    forgotten one still on it — for up to a full TTL after a write that
    already succeeded, which reads as "the deck ignored me".

    GATED on the node being a non-local entry, matching the predicate the
    assembly itself walks by (``remote_engine_declarations`` skips
    ``agent_kind == "local"``): a local declaration cannot change anything
    the observer holds, and invalidating clears every node's BACKOFF too —
    so an ungated call would re-probe powered-off boxes (5 s transport
    timeout each) on the next request, for an edit that provably cannot
    affect them."""
    if node["agent_kind"] == "local":
        return
    observer = deck.get("remote_observer")
    if observer is not None:
        observer.invalidate()


def _shape_error(msg: str, *loc: str) -> RequestValidationError:
    """Wrap a body-shape defect as a RequestValidationError (module
    docstring: never the bare ValueError handler for a request body) —
    `loc` mirrors pydantic's own error-location tuple convention."""
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body", *loc), "msg": msg}])


@router.post("/{node_id}/engines")
def add_engine(node_id: str, body: dict, request: Request) -> dict:
    """Declare a new engine on `node_id` (E1 Task 5: node-scoped —
    `/nodes/local/engines` is unchanged behavior, "local" is just an id).
    404 for an unknown node. 422 for a shape defect — `remote` is passed to
    `validate_engines` matching the TARGET node's `agent_kind`, so a
    node-agent target's "kind not remote_capable" refusal is also caught
    here and goes through the same redacting handler as every other shape
    defect (see this router's module docstring), rather than falling
    through to NodeStore.update's own re-validation and its plain-string
    422. 409 for a resource already declared — checked AFTER shape
    validation (NodeStore.add's own order: full validate, then
    duplicate-identity check). A node-agent target additionally needs
    agent operability (address + credential) — that check has no shape to
    diagnose here (it depends on the credential sidecar, not the request
    body) and is left to NodeStore.update's own ValueError, same posture
    as `create_node`'s control:"swap" prereqs."""
    deck = request.app.state.deck
    store = deck["node_store"]
    node = _node_or_404(deck, node_id)
    try:
        validate_engines([body], remote=node["agent_kind"] == "node-agent")
    except ValueError as exc:
        raise _shape_error(str(exc)) from exc
    resource = body["resource"]
    engines = list(node.get("engines", []))
    if any(e["resource"] == resource for e in engines):
        raise HTTPException(status_code=409,
                            detail=f"engine {resource!r} already exists")
    entry = store.update(node_id, {"engines": engines + [body]})
    _reobserve(deck, node)
    log_event(deck["events_path"], "engine-added",
              {"node": node_id, "resource": resource, "kind": body["kind"]})
    return next(e for e in entry["engines"] if e["resource"] == resource)


@router.put("/{node_id}/engines/{resource}")
def update_engine(node_id: str, resource: str, body: dict, request: Request) -> dict:
    """Full-entry replace, node-scoped (E1 Task 5 — see add_engine's
    docstring for the `remote`/404/409-ordering rationale, identical here).
    `resource` in the body must equal the path — a
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
    own three-write sequence, and deliberately so).

    The intent-forget below is keyed through `node_key(node_id, resource)`
    (sglang-omni Task 6), not the E1 `local_key` it used to be gated on:
    the record it must invalidate is the one for THIS node's resource, and
    `node_key("local", r)` is `local_key(r)` exactly, so the local path is
    byte-identical. It became reachable for a non-local node when Task 7
    flipped the first `remote_capable` kind, and was already correct BY KEY
    for it — a remote kind change forgets THAT node's record, never the
    local resource of the same name. (Note the asymmetry with
    `forget_engine`'s policy write below, which is gated to the local node
    per ruling R7: the INTENT store has a node dimension, PolicyStore does
    not.)"""
    deck = request.app.state.deck
    store = deck["node_store"]
    node = _node_or_404(deck, node_id)
    if body.get("resource") != resource:
        raise _shape_error(
            f"resource {body.get('resource')!r} must match the path "
            f"{resource!r} — rename is refused; forget and re-add instead",
            "resource")
    engines = list(node.get("engines", []))
    existing = next((e for e in engines if e["resource"] == resource), None)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"unknown engine {resource!r}")
    try:
        validate_engines([body], remote=node["agent_kind"] == "node-agent")
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
        deck["intent_store"].forget(node_key(node_id, resource))
    new_engines = [body if e["resource"] == resource else e for e in engines]
    entry = store.update(node_id, {"engines": new_engines})
    _reobserve(deck, node)
    log_event(deck["events_path"], "engine-updated",
              {"node": node_id, "resource": resource})
    return next(e for e in entry["engines"] if e["resource"] == resource)


@router.delete("/{node_id}/engines/{resource}")
def forget_engine(node_id: str, resource: str, request: Request) -> dict:
    """Forget (spec §6.2 / coexistence ruling — module docstring):
    bookkeeping only, node-scoped (E1 Task 5). Drops the declaration entry,
    the intent record, and the stored policy row — nothing else. Never
    calls the engine: no client lookup anywhere in this function. 404 when
    the node OR the engine is unknown.

    The intent-forget below is node-keyed since sglang-omni Task 6
    (`node_key(node_id, resource)`, same change as update_engine's — see
    that docstring): it forgets THIS node's record, and
    `node_key("local", r)` is `local_key(r)` exactly, so the local path is
    byte-identical.

    `policy_store.forget(resource)` below is node-BLIND by key, and correct
    for every node since sglang-omni Task 9 (controller ruling R10). Two
    things landed together to make that true, and neither works without the
    other:

    * policy rows are now seeded from EVERY entry's engines[]
      (app.policy.declared_defaults), so a remote engine really has a row of
      its own to forget — under R7 it had none, which is why that ruling
      GATED this call on the local node rather than node-scoping it;
    * a resource name is now unique across the whole deck
      (app.node_store's `_require_unique_resources`, refusing at the
      declaration boundary, `_heal_unique_resources` healing a hand-edit),
      so a bare resource key resolves to exactly ONE declaration. "The
      unrelated LOCAL row of the same name" — the collision R7's guard
      existed to avoid — is no longer a state the registry can hold.

    PolicyStore itself keeps its flat `{resource: {...}}` rows and gains no
    node dimension: R10 buys the unambiguity at the boundary the names enter
    through instead, which is one gate rather than a keying migration
    through the store, its declared-defaults source, `app.arbiter`'s
    `policy.get(resource)` lookup, the policy router and the UI.
    tests/test_api.py's ...pops_its_own_policy_row and
    ...refused_naming_the_owner are the pair that proves it (the first
    records the handover from R7's tripwire)."""
    deck = request.app.state.deck
    store = deck["node_store"]
    node = _node_or_404(deck, node_id)
    engines = list(node.get("engines", []))
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
    store.update(node_id, {"engines": new_engines})
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
    #
    # Node-keyed (sglang-omni Task 6): the E1 `node_id == "local"` guard
    # that used to wrap this call existed only because `local_key` always
    # spelled "local/<resource>", so forgetting a REMOTE node's resource
    # named e.g. "foo" would have forgotten the unrelated LOCAL "foo"
    # record. `node_key` removes the collision at its source rather than
    # gating around it, and answers `local_key(resource)` verbatim for the
    # local node — so this line is unchanged for every reachable caller
    # today and correct for the remote ones Task 7 unlocks.
    deck["intent_store"].forget(node_key(node_id, resource))
    # Every node (ruling R10, sglang-omni Task 9 — R7's local-only guard is
    # lifted): the row belongs to THIS resource wherever it was declared,
    # because a resource name can only be declared on one node deck-wide.
    # See this function's docstring for the two changes that make the bare
    # key unambiguous.
    deck["policy_store"].forget(resource)
    _reobserve(deck, node)
    log_event(deck["events_path"], "engine-removed",
              {"node": node_id, "resource": resource})
    return {"status": "ok"}


@kinds_router.get("/engine-kinds")
def list_engine_kinds() -> dict:
    """The UI's kind picker source (spec §5): every known kind's connection
    schema (field -> required), WHERE it may run (`remote_capable` — may it
    be declared on a node-agent entry; `local_capable` — may it be declared
    on the local one), and its human-initiated verb vocabulary, served from
    app.engine_kinds — the one module allowed to know engine names (spec
    §8) — so the picker never bakes a kind name into the UI.

    Both run-location flags are served, not just the remote one (Task 7 fix
    round 1): they are exactly what `validate_engines` enforces, so a picker
    that filters on them offers only kinds the write gate will accept for
    the node being edited, rather than surfacing a 422 after the fact.
    `ui/src/model/engineForm.ts` reads only `connection`/`human_verbs` off
    this payload today, so both are additive — no UI change is needed for it
    to keep working, and the filtering is available when the editor gains
    non-local nodes."""
    kinds = []
    for kind in sorted(KNOWN_KINDS):
        spec = KNOWN_KINDS[kind]
        kinds.append({
            "kind": kind,
            "connection": {field: {"required": required}
                          for field, required in spec["connection"].items()},
            "remote_capable": spec["remote_capable"],
            "local_capable": spec["local_capable"],
            # Whether a released resource of this kind comes BACK by itself.
            # This is what makes idle_ttl either free (lemonade: the next
            # request reloads it, and an idle resident model burns ~70 W for
            # nothing) or one-way (everything else: the operator returns to a
            # gone engine, ~4 min to rebuild for sglang-omni, GF4). Served
            # rather than inferred so no component learns a kind name
            # (spec §8) -- app/engine_kinds.py's per-kind `demand()`.
            "demand": ENGINE_KINDS[kind].demand(),
            "human_verbs": sorted(ENGINE_KINDS[kind].human_verbs()),
            # Whether a nonzero idle_ttl on this kind DOES ANYTHING. hipfire's
            # `idle_action` is unconditionally None and its `arbiter_verbs()`
            # is empty (app/engine_kinds.py's _HipfireAdapter: "No arbiter
            # verb -> no idle rule either: park stays human-only. Structural
            # omission made explicit") — a nonzero TTL on it is a no-op, and
            # without this flag the UI's ttlConsequence had no way to say so
            # (it rendered the false "reload is MANUAL" sentence instead,
            # which implies a rule that fires and simply never reloads).
            # Derived from `arbiter_verbs()` being non-empty rather than a
            # new adapter method: verified against all four adapters that
            # exactly the kinds with a real idle_action also have a
            # non-empty arbiter_verbs() (lemonade/comfyui/sglang-omni yes,
            # hipfire no) — see tests/test_api.py's
            # test_engine_kinds_serves_idle_release_... for the per-line
            # citations this comment summarizes.
            "idle_release": bool(ENGINE_KINDS[kind].arbiter_verbs()),
        })
    return {"kinds": kinds}
