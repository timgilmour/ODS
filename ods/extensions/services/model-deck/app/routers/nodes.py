"""Nodes router — the registry's CRUD + the test-connection probe.

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
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.engines import EngineError
from app.events import log_event

router = APIRouter(prefix="/nodes", tags=["nodes"])


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
