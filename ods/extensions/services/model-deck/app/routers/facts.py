"""Facts router — resolved characteristics and drift, read-mostly.

``GET /facts`` merges derived and declared facts with provenance intact.
``PUT /facts/declared/{key}`` is the only write, and the DeclaredStore
allowlist is what keeps it from becoming a way to duplicate a derivable
fact (a rejected field raises ValueError -> 422 via the app-level handler).
The key itself is validated too — it must start with "model/" or "engine/"
with a non-empty, non-trailing-slash remainder, and the fields body must be
non-empty — so an empty key, "model/m/", "junk/x", or {} can never durably
materialize a phantom entry.

``GET /facts/drift`` compares facts that should agree, per key. What fires
in THIS increment, and why:

* ``max_model_len`` (live-served context vs. the checkpoint's
  ``max_position_embeddings``) fires now, with no gateway I/O at all — its
  runtime side is this SAME key's own already-derived ``max_model_len_live``
  fact (characteristics.json, populated by the watcher's live-derive pass
  from /v1/models), fed straight back into `detect_drift`.
* ``max_input_tokens`` (what the gateway advertises vs. what the engine
  actually serves) fires now, sourced from LiteLLM's ``/model/info``. The
  join is on the route's RESOLVED model (``litellm_params.model``, with the
  "openai/" prefix stripped — mirrors ``app.state``'s handling of the same
  field) — never ``model_name``, which is a route ALIAS ("default",
  "hipfire", "*") and will never match a checkpoint-shaped facts key.
* ``quantization`` (severity ``crash``) does NOT fire through this endpoint
  yet — it is implemented and unit-tested in ``app.facts``, but needs
  profile flags, which only become readable starting Plan C1. The rule
  exists now so C1 turns it on rather than inventing it.

Keys contain a slash (``model/Qwen3.6-...``), hence the ``:path`` converter.
"""

from fastapi import APIRouter, Request

from app.engines import EngineError
from app.facts import detect_drift, resolve_facts

router = APIRouter(tags=["facts"])

# Mirrors app.state._OPENAI_PREFIX: litellm_params.model is always
# "openai/<resolved id>" in the real /model/info response.
_OPENAI_PREFIX = "openai/"

# A declared key's kind — mirrors app.characteristics' "<kind>/<id>" shape.
# Engine keys legitimately contain a second slash (e.g. "engine/boxa/vllm"
# — node/engine), so only an EMPTY remainder or a trailing slash is rejected,
# never an interior one.
_KEY_PREFIXES = ("model/", "engine/")


def _validate_declared_key(key: str) -> None:
    """Reject a key shape that would durably materialize a phantom entry:
    empty, unprefixed junk, or a trailing slash. Raises ValueError, which
    the app-level handler turns into 422 (see the module docstring)."""
    if not key or key.endswith("/"):
        raise ValueError(
            f"declared key {key!r} must not be empty or end with a slash")
    if not key.startswith(_KEY_PREFIXES):
        raise ValueError(
            f"declared key {key!r} must start with 'model/' or 'engine/'")


@router.get("/facts")
def get_facts(request: Request) -> dict:
    deck = request.app.state.deck
    derived = deck["characteristics_store"].get()
    declared = deck["declared_store"].get()

    keys = set(derived) | set(declared)
    return {key: resolve_facts(derived.get(key, {}), declared.get(key, {})) for key in sorted(keys)}


@router.put("/facts/declared/{key:path}")
def put_declared(key: str, fields: dict, request: Request) -> dict:
    _validate_declared_key(key)
    if not fields:
        raise ValueError("PUT /facts/declared requires a non-empty fields body")
    deck = request.app.state.deck
    deck["declared_store"].put(key, fields)
    return {"key": key, "declared": deck["declared_store"].entry(key)}


@router.get("/facts/drift")
def get_drift(request: Request) -> dict:
    """Per-key drift. Absent facts mean 'cannot check', never 'mismatch'.

    See the module docstring for exactly which rules fire in this
    increment and why.
    """
    deck = request.app.state.deck
    derived = deck["characteristics_store"].get()
    declared = deck["declared_store"].get()

    runtime_by_model = _gateway_runtime(deck)

    report = {}
    for key in sorted(set(derived) | set(declared)):
        facts = resolve_facts(derived.get(key, {}), declared.get(key, {}))
        model_id = key.split("/", 1)[-1]
        runtime = dict(runtime_by_model.get(model_id, {}))

        # max_model_len needs no gateway I/O: it's a live-vs-checkpoint
        # check within this SAME key's own already-derived facts.
        live_len = facts.get("max_model_len_live")
        if live_len is not None:
            runtime["max_model_len"] = live_len["value"]

        drift = detect_drift(facts, runtime)
        if drift:
            report[key] = drift
    return report


def _gateway_runtime(deck: dict) -> dict[str, dict]:
    """What the gateway advertises per RESOLVED model id, or {} when
    litellm is down — an unreachable gateway is not evidence of a
    mismatch.

    Joined on ``litellm_params.model`` (the "openai/"-prefixed model a
    route actually resolves to), NEVER ``model_name`` — that field is the
    route ALIAS ("default", "hipfire", "*"), not a checkpoint id, and a
    join keyed by it would silently match nothing in production.
    """
    try:
        entries = deck["litellm"].model_info()
    except EngineError:
        return {}

    runtime: dict[str, dict] = {}
    for entry in entries:
        resolved = (entry.get("litellm_params") or {}).get("model")
        if not resolved:
            continue
        model_id = resolved.removeprefix(_OPENAI_PREFIX)
        info = entry.get("model_info") or {}
        if info.get("max_input_tokens") is not None:
            runtime[model_id] = {"max_input_tokens": info["max_input_tokens"]}
    return runtime
