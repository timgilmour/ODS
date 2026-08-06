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

import time

from app.engines import BusyError, EngineError, GuardError
from app.engines.spark import boot_in_flight

_LOCAL_NODE = "local"

# Public so writers (app.arbiter, app.routers.control) and readers
# (app.arbiter's reconcile pass) name it from one place instead of
# re-typing the literal — actuation and observation can never disagree on
# the key.
LOCAL_LEMONADE_KEY = f"{_LOCAL_NODE}/lemonade"

# Spark is a single-slot serving node: one resource, always this key. Public
# so writers (app.routers.spark) and readers (app.arbiter) name it from one
# place instead of re-typing the literal.
SPARK_SLOT_KEY = "sparky/slot0"


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
            # "loading" = a load is in flight (LemonadeClient.load_in_flight,
            # app/state.py's _snapshot_lemonade). Neither loaded nor dead;
            # acting on it restarts a model that is already mid-load.
            transitioning=lemonade["state"] == "loading",
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
            SPARK_SLOT_KEY: {
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
        SPARK_SLOT_KEY: {
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
    SPARK_SLOT_KEY: "spark",
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


# ---------------------------------------------------------------------------
# Spark observation: the node payload -> this module's vocabulary, cached.
# ---------------------------------------------------------------------------

# SparkClient.status() costs TWO node-agent requests, and watch_interval
# defaults to 2.0 s. When sparky is off — its normal state most of the time —
# each of those blocks on a 5 s httpx timeout, which would stretch the
# arbiter's cadence from 2 s to ~12 s exactly when nothing is wrong. So a
# short TTL on success, and a growing backoff on failure.
SPARK_OBSERVE_TTL_S = 10.0
SPARK_BACKOFF_BASE_S = 15.0
SPARK_BACKOFF_MAX_S = 300.0

_UNREACHABLE_SPARK = {
    "profile": None, "serving": None, "reachable": False, "swap_in_progress": False,
}


def translate_spark_status(payload: dict) -> dict:
    """The ONE translation of ``SparkClient.status()`` into this module's
    vocabulary. It exists because the node payload
    (``{"profiles", "swap_status", "serving"}``) is not the shape
    ``observe_spark`` consumes, and a second copy of this mapping is how the
    next drift bug gets in.

    Identity is the PROFILE the node last swapped to, not the served model
    name: mm27b serves under ``--served-model-name aeon``, so comparing
    served names would report permanent drift for a correct placement.
    """
    swap_status = payload.get("swap_status") or {}
    return {
        "profile": swap_status.get("profile"),
        "serving": payload.get("serving"),
        "reachable": True,
        "swap_in_progress": boot_in_flight(payload),
    }


class SparkObserver:
    """Cached, backed-off access to the spark node's observation.

    Shared by the watcher and the HTTP paths (one instance in the deck), so
    a tick and a ``GET /api/state`` in the same second cost one probe, not
    three. An engine failure reads as *unreachable*, never as "nothing
    loaded" — we failed to look, which is not the same as looking and seeing
    nothing, and the difference decides whether the reconciler acts.

    Errors are parked rather than raised or logged: the watcher owns the
    audit log and takes them on its next tick (``take_error``), so a probe
    that happened to run on an HTTP thread still gets reported exactly once.
    """

    def __init__(
        self,
        spark_fn,
        *,
        ttl_s: float = SPARK_OBSERVE_TTL_S,
        backoff_base_s: float = SPARK_BACKOFF_BASE_S,
        backoff_max_s: float = SPARK_BACKOFF_MAX_S,
        clock=time.monotonic,
    ) -> None:
        # A callable, not a client: the deck's "spark" entry is swapped by
        # tests after construction, and late binding keeps one observer
        # correct for the process lifetime.
        self._spark_fn = spark_fn
        self._ttl_s = ttl_s
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._clock = clock
        self._cached: dict | None = None
        self._next_probe_at = 0.0
        self._failures = 0
        self._error: str | None = None

    def status(self) -> dict | None:
        """The spark observation, or None when no spark is configured at all
        (``observe_spark(None)`` then emits no key — an undeclared resource
        must not appear as a phantom failure)."""
        spark = self._spark_fn()
        if spark is None:
            return None

        now = self._clock()
        if self._cached is not None and now < self._next_probe_at:
            return self._cached

        try:
            payload = spark.status()
        except (EngineError, GuardError, BusyError) as exc:
            self._failures += 1
            self._error = str(exc)
            self._cached = dict(_UNREACHABLE_SPARK)
            self._next_probe_at = now + min(
                self._backoff_base_s * 2 ** (self._failures - 1), self._backoff_max_s
            )
            return self._cached

        self._failures = 0
        self._cached = translate_spark_status(payload)
        self._next_probe_at = now + self._ttl_s
        return self._cached

    def invalidate(self) -> None:
        """Drop the cache — call after acting on the node (a swap), whose
        whole purpose is to change what this caches."""
        self._cached = None
        self._next_probe_at = 0.0
        self._failures = 0

    def take_error(self) -> str | None:
        """The last probe failure, cleared by reading it."""
        error, self._error = self._error, None
        return error
