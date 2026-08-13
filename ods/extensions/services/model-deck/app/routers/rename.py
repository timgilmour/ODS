"""Rename router — the alias -> identity migration PLANNER's HTTP surface
(Plan C2, Task 10). ``POST /api/rename/plan`` is the only route.

Read-only, by construction: it gathers today's real vLLM profiles (the same
status + get_compose + import_compose machinery as
``app.routers.settings.adopt``, Task 5) and today's real litellm routes
(``deck["litellm"].route_table()``), hands them straight to
``app.rename.plan_rename`` -- a PURE function, no I/O of its own -- and
returns whatever comes back. Nothing here writes to the settings store, the
characteristics store, or intent, so calling this route any number of times
in a row returns the identical plan every time; that idempotence is the
whole point of shipping a PLANNER before an executor. Executing a rename
(regenerating a profile's --served-model-name, regenerating litellm routes,
notifying a pinned client like OMP) is deliberately OUT of scope here -- see
app.rename's module docstring -- and would be its own plan with its own
gate.

Per-profile isolation mirrors ``app.routers.settings.adopt`` (Task 5's fixed
form): a profile whose compose fetch or parse fails, or that has no /model
mount, or that carries no --served-model-name at all, is simply left out of
the gathered profiles -- there is nothing to plan a rename around for it --
rather than failing the whole request. Only the resolved swap node having no
live client (no control:"swap" node declared, or a known one that is not
currently operable) is a whole-request precondition, 503, matching
``app.routers.spark._single_swap_node_id``'s guard exactly (including its
409 when more than one swap node is declared -- never guess which one the
caller meant, [[literal-declared-inputs]]).
"""

from fastapi import APIRouter, HTTPException, Request

from app.compose_import import import_compose
from app.engines import EngineError
from app.rename import plan_rename
from app.routers.spark import _single_swap_node_id

router = APIRouter(prefix="/rename", tags=["rename"])


def _gather_vllm_profiles(spark) -> dict:
    """``{<profile>: {"served_model_name": [...], "identity": str}}`` for
    every real vLLM profile on the node right now -- the exact shape
    ``plan_rename`` expects. A profile that isn't vLLM, whose compose can't
    be fetched or parsed, that has no /model mount (``identity`` is
    ``None``), or that declares no --served-model-name at all has nothing a
    rename plan could act on and is skipped, silently, the same way
    ``app.routers.settings.adopt`` reports (but does not fail on) a bad
    profile -- one unreadable profile must not block a plan for every
    OTHER profile that parses cleanly."""
    profiles: dict = {}
    for meta in spark.status()["profiles"]:
        if meta["engine"] != "vllm":
            continue
        try:
            imported = import_compose(spark.get_compose(meta["name"]))
        except (ValueError, EngineError):
            continue

        identity = imported["identity"]
        if identity is None:
            continue

        served = imported["args"].get("served-model-name")
        if served is None:
            continue
        # parse_argline's singleton-collapse axis (app.argline module
        # docstring): a lone --served-model-name value comes back as a bare
        # str, not a list of one -- normalize both shapes to a list here so
        # plan_rename never has to care which one it got.
        if isinstance(served, str):
            served = [served]

        profiles[meta["name"]] = {"served_model_name": served, "identity": identity}
    return profiles


@router.post("/plan")
def rename_plan(request: Request, body: dict | None = None) -> dict:
    """Gather real profiles + routes, return ``plan_rename``'s output
    unchanged. ``body`` (and its ``client_pins`` key) default to ``{}`` --
    the deck cannot see a pinned client's own config (OMP, say), so the
    caller supplies whatever pins the runbook already knows about; an empty
    or absent body just means the caller has none to report."""
    deck = request.app.state.deck
    node_id = _single_swap_node_id(deck)
    spark = deck["node_clients"].client_for(node_id)
    if spark is None:
        raise HTTPException(status_code=503,
                            detail="spark engine is not configured")

    profiles = _gather_vllm_profiles(spark)
    routes = deck["litellm"].route_table()
    client_pins = (body or {}).get("client_pins") or {}

    return plan_rename(profiles, routes, client_pins)
