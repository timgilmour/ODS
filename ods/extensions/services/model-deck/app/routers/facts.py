"""Facts router — resolved characteristics and drift, read-mostly.

``GET /facts`` merges derived and declared facts with provenance intact.
``PUT /facts/declared/{key}`` is the only write, and the DeclaredStore
allowlist is what keeps it from becoming a way to duplicate a derivable
fact (a rejected field raises ValueError -> 422 via the app-level handler).

``GET /facts/drift`` compares facts that should agree. In THIS increment its
inputs are the gateway route table and the live /v1/models surface, so the
context rules fire and the `quantization` rule does not — profile flags
only become readable in Plan C1, when settings land. The rule is
implemented and unit-tested now so that C1 turns it on rather than
inventing it.

Keys contain a slash (``model/Qwen3.6-...``), hence the ``:path`` converter.
"""

from fastapi import APIRouter, Request

from app.engines import EngineError
from app.facts import detect_drift, resolve_facts

router = APIRouter(tags=["facts"])


@router.get("/facts")
def get_facts(request: Request) -> dict:
    deck = request.app.state.deck
    derived = deck["characteristics_store"].get()
    declared = deck["declared_store"].get()

    keys = set(derived) | set(declared)
    return {key: resolve_facts(derived.get(key, {}), declared.get(key, {})) for key in sorted(keys)}


@router.put("/facts/declared/{key:path}")
def put_declared(key: str, fields: dict, request: Request) -> dict:
    deck = request.app.state.deck
    deck["declared_store"].put(key, fields)
    return {"key": key, "declared": deck["declared_store"].entry(key)}


@router.get("/facts/drift")
def get_drift(request: Request) -> dict:
    """Per-key drift. Absent facts mean 'cannot check', never 'mismatch'."""
    deck = request.app.state.deck
    derived = deck["characteristics_store"].get()
    declared = deck["declared_store"].get()

    runtime_by_model = _gateway_runtime(deck)

    report = {}
    for key in sorted(set(derived) | set(declared)):
        facts = resolve_facts(derived.get(key, {}), declared.get(key, {}))
        model_id = key.split("/", 1)[-1]
        drift = detect_drift(facts, runtime_by_model.get(model_id, {}))
        if drift:
            report[key] = drift
    return report


def _gateway_runtime(deck: dict) -> dict[str, dict]:
    """What the gateway advertises per model, or {} when litellm is down —
    an unreachable gateway is not evidence of a mismatch."""
    try:
        entries = deck["litellm"].model_info()
    except EngineError:
        return {}

    runtime = {}
    for entry in entries:
        name = entry.get("model_name")
        info = entry.get("model_info") or {}
        if name and info.get("max_input_tokens") is not None:
            runtime[name] = {"max_input_tokens": info["max_input_tokens"]}
    return runtime
