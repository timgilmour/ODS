"""Model Deck observation adapter — per-engine vocabularies -> one shape.

Every engine describes itself differently: lemonade says loaded/unloaded,
hipfire says running/stopped, comfyui says busy/idle, and all of them say
"unknown" when app.state caught an EngineError. Downstream, app.lifecycle
wants exactly one shape: ``{"reachable": bool, "loaded": bool, "model": str|None}``.

This module is the ONLY place the per-engine vocabulary appears. That is
the point: adding an engine means adding a mapping here, not editing
status derivation, reconciliation, or the API. Pure functions, no I/O — the
caller passes in an already-built World snapshot.

Two mappings worth stating explicitly because they are not obvious:

* **"unknown" means unreachable, not unloaded.** app.state degrades a tenant
  to "unknown" only when its status call raised — we did not observe
  "nothing loaded", we failed to observe at all. Calling that "not loaded"
  would let the reconciler restore something that may already be running.
* **ComfyUI's "idle" still counts as loaded.** It holds VRAM between jobs;
  idle describes its queue, not its residency.
"""

_LOCAL_NODE = "local"

# Spark is a single-slot serving node: one resource, always this key.
_SPARK_SLOT_KEY = "sparky/slot0"


def observe_local(world: dict) -> dict[str, dict]:
    """Map a World snapshot's tenants to observation records."""
    tenants = world["tenants"]

    lemonade = tenants["lemonade"]
    hipfire = tenants["hipfire"]
    comfy = tenants["comfyui"]

    return {
        f"{_LOCAL_NODE}/lemonade": _record(
            unknown=lemonade["state"] == "unknown",
            loaded=lemonade["state"] == "loaded",
            model=lemonade.get("model"),
        ),
        f"{_LOCAL_NODE}/hipfire": _record(
            unknown=hipfire["state"] == "unknown",
            loaded=hipfire["state"] == "running",
            model=hipfire.get("model"),
            # "loading" = container up, health not yet 200. Neither loaded
            # nor dead; acting on it restarts a model that is already on its
            # way up (HipfireClient.status, app/engines/hipfire.py:100-107).
            transitioning=hipfire["state"] == "loading",
        ),
        f"{_LOCAL_NODE}/comfyui": _record(
            unknown=comfy["state"] == "unknown",
            loaded=comfy["state"] in ("busy", "idle"),
            model=None,
        ),
    }


def observe_spark(spark_status: dict | None) -> dict[str, dict]:
    """Map the spark node's status to its single slot resource.

    ``None`` means no spark is configured at all, which emits no key —
    a resource nobody declared must not appear as a phantom failure.
    """
    if spark_status is None:
        return {}

    if not spark_status.get("reachable", False):
        return {
            _SPARK_SLOT_KEY: {
                "reachable": False, "loaded": False, "model": None, "transitioning": False,
            }
        }

    serving = spark_status.get("serving") or {}
    # Identity is the PROFILE, not the served model name: swap takes a
    # profile, so that is the thing intent can be compared against. mm27b
    # serves under --served-model-name aeon; comparing served names would
    # report permanent drift for a perfectly correct placement.
    profile = spark_status.get("profile")
    return {
        _SPARK_SLOT_KEY: {
            "reachable": True,
            "loaded": serving.get("model") is not None,
            "model": profile if serving.get("model") is not None else None,
            "transitioning": bool(spark_status.get("swap_in_progress")),
        }
    }


# Which engine owns each resource key. Used by restore dispatch and adopt;
# a new engine adds a line here rather than editing either caller.
_ENGINE_BY_KEY = {
    "local/lemonade": "lemonade",
    "local/hipfire": "hipfire",
    "local/comfyui": "comfyui",
    _SPARK_SLOT_KEY: "spark",
}


def engine_for(key: str) -> str | None:
    """The engine that owns `key`, or None if the key is unknown."""
    return _ENGINE_BY_KEY.get(key)


def merge_observations(*maps: dict[str, dict]) -> dict[str, dict]:
    """Combine per-node observation maps into one flat mapping."""
    merged: dict[str, dict] = {}
    for m in maps:
        merged.update(m)
    return merged


def _record(
    *, unknown: bool, loaded: bool, model: str | None, transitioning: bool = False
) -> dict:
    if unknown:
        return {"reachable": False, "loaded": False, "model": None, "transitioning": False}
    return {
        "reachable": True,
        "loaded": loaded,
        "model": model if loaded else None,
        "transitioning": transitioning,
    }
