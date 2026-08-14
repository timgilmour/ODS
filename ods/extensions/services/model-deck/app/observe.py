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

def local_key(resource: str) -> str:
    """The intent/observation key for a DECLARED local resource
    (``local/<resource>``) — the per-resource generalization of the old
    ``LOCAL_LEMONADE_KEY`` constant (E1 Task 6: that name is gone; every
    former use now calls this instead). Writers (app.engine_kinds'
    actuator methods, app.sets, app.routers.control) and readers
    (app.arbiter's reconcile pass, ``observe_local`` below) derive the key
    from HERE so actuation and observation can never disagree on it — same
    rationale ``LOCAL_HIPFIRE_KEY`` below already followed for hipfire
    specifically, generalized past a single hardcoded name."""
    return f"{_LOCAL_NODE}/{resource}"


# Same rationale as local_key above, kept as a named constant for hipfire
# specifically: app.sets and app.routers.control still address hipfire by
# its one legacy resource name directly (E1 Task 6 scope: only
# app/arbiter.py's LOCAL_LEMONADE_KEY uses generalize this task — control.py's
# route generalization is Task 7's).
LOCAL_HIPFIRE_KEY = local_key("hipfire")


def slot_key(node_id: str) -> str:
    """The serving-slot resource key for a swap node. "slot0" is the
    documented single-slot convention (multi-slot is out of scope, design
    §4). Writers (app.routers.serving) and readers (app.arbiter,
    app.routers.build_observations) derive the key from HERE so actuation
    and observation can never disagree on it."""
    return f"{node_id}/slot0"


def observe_local(world: dict) -> dict[str, dict]:
    """Map a World snapshot's tenants to observation records, one per
    DECLARED resource (E1 generalization, Task 3: a resource nobody
    declared emits no key here — no phantom "unknown" entry, spec §1's
    absence-is-representable rule — rather than the pre-E1 fixed three).

    Field mapping is still per KIND (each REAL World.snapshot tenant
    carries `engine`, stamped there) — a second lemonade-kind resource maps
    exactly like the first. This dispatch is a disclosed, small residue of
    engine-kind-name literals outside app.engine_kinds (see that module's
    docstring): Task 3's brief scoped app/observe.py out, but leaving this
    function's old hardcoded `tenants["lemonade"]`/`tenants["hipfire"]`/
    `tenants["comfyui"]` reads in place would KeyError on every
    `GET /api/state` call once World stopped guaranteeing those three keys
    — a real, unavoidable break, not a style choice. E1 Task 6 relocated
    the LOCAL_LEMONADE_KEY constant this function used to define (now
    ``local_key(resource)``, above) but deliberately left this per-kind
    if/elif dispatch itself alone — out of that task's scope (arbiter.py's
    watcher execution/pending/restore paths), still a disclosed residue.

    `tenant["engine"]` is read unconditionally — the transitional
    `tenant.get("engine", resource)` fallback this function used to lean on
    (for hand-built world dicts across app/arbiter.py's and
    tests/test_arbiter.py's `decide()` pure-function fixtures, which predated
    the `engine` field) is gone: Task 5 backfilled `engine` into every one of
    those fixtures (plus the couple of other hand-built worlds elsewhere in
    tests/ that also route through this function), so a resource-name guess
    is no longer needed — every tenant this function ever sees now carries
    its own kind explicitly, same as production."""
    out: dict[str, dict] = {}
    for resource, tenant in world["tenants"].items():
        kind = tenant["engine"]
        state = tenant["state"]
        if kind == "lemonade":
            record = _record(
                unknown=state == "unknown",
                loaded=state == "loaded",
                model=tenant.get("model"),
                # "loading" = a load is in flight (LemonadeClient.load_in_flight,
                # app/engine_kinds.py's lemonade adapter). Neither loaded nor
                # dead; acting on it restarts a model that is already mid-load.
                transitioning=state == "loading",
            )
        elif kind == "hipfire":
            record = _record(
                unknown=state == "unknown",
                loaded=state == "running",
                model=tenant.get("model"),
                # "loading" = container up, health not yet 200. Neither loaded
                # nor dead; acting on it restarts a model that is already on its
                # way up (HipfireClient.status, app/engines/hipfire.py:100-107).
                transitioning=state == "loading",
            )
        elif kind == "comfyui":
            record = _record(
                unknown=state == "unknown",
                loaded=state in ("busy", "idle"),
                model=None,
            )
        else:
            # Not reachable from a REAL World.snapshot (NodeStore validates
            # `kind` against engine_kinds.KNOWN_KINDS before an entry can
            # ever land in the declaration one was built from) nor from any
            # existing hand-built world dict (every one in this codebase
            # names its tenants lemonade/comfyui/hipfire) — only a future
            # fourth kind or a fixture using some other resource name
            # without an explicit "engine" key would reach this.
            raise ValueError(f"unhandled engine kind {kind!r}")
        out[local_key(resource)] = record
    return out


def observe_spark(spark_status: dict | None, node_id: str) -> dict[str, dict]:
    """Map one swap node's status to its single slot resource.

    ``None`` means the node is not observable at all (no observer, no
    operable client), which emits no key — a resource nobody declared must
    not appear as a phantom failure.
    """
    if spark_status is None:
        return {}

    key = slot_key(node_id)
    if not spark_status.get("reachable", False):
        return {
            key: {
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
        key: {
            "reachable": True,
            "loaded": serving.get("model") is not None,
            "model": profile if serving.get("model") is not None else None,
            "transitioning": bool(spark_status.get("swap_in_progress")),
        }
    }


# Which LOCAL engine owns each resource key, for a caller with no live
# declaration to consult (every caller before this fix). This static table
# is the exact resource==kind-name-coincidence bug
# test_adopt_a_swap_nodes_slot's docstring (N1 T12) already fixed for the
# swap-node half of this same function: it only ever matched a resource
# literally named lemonade/comfyui/hipfire, so a genuinely DECLARED (E1)
# resource under some other name (e.g. "gguf-a", kind "lemonade") silently
# 404'd out of the lifecycle router's adopt route with "no engine owns
# ...". `local_kinds` below (E1 T13 fix) is the local-side counterpart to
# `swap_node_ids` — a PREPARED resource->kind mapping the caller reads live
# off node_store (mirrors app.routers.control._declared_kind /
# app.routers.sets._declared_kinds) — checked FIRST when supplied. This
# table remains the fallback for a caller that still doesn't thread one
# through, so the three tests pinning its literal behavior stay exactly
# correct.
_LOCAL_ENGINE_BY_KEY = {
    local_key("lemonade"): "lemonade",
    LOCAL_HIPFIRE_KEY: "hipfire",
    local_key("comfyui"): "comfyui",
}


def engine_for(
    key: str,
    swap_node_ids: frozenset[str] | set[str] = frozenset(),
    local_kinds: dict[str, str] | None = None,
) -> str | None:
    """The engine that owns `key`, or None if the key is unknown.

    `local_kinds` (E1 T13, optional): resource -> declared kind for the
    LOCAL node, checked first when supplied — see this module's own
    `_LOCAL_ENGINE_BY_KEY` docstring for why. None (every pre-E1-T13
    caller) skips straight to the static table below, unchanged.

    Failing that, a ``<node>/slot0`` key maps to ``"spark"`` iff ``<node>``
    is in `swap_node_ids` — a PREPARED id-set, not the store, so this stays
    a pure function and each caller decides its own registry read (design
    §4)."""
    if local_kinds is not None:
        resource = key.removeprefix(f"{_LOCAL_NODE}/")
        if resource != key and resource in local_kinds:
            return local_kinds[resource]
    engine = _LOCAL_ENGINE_BY_KEY.get(key)
    if engine is not None:
        return engine
    node, sep, resource = key.partition("/")
    if sep and resource == "slot0" and node in swap_node_ids:
        return "spark"
    return None


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
# defaults to 2.0 s. When a swap node is off — its normal state most of the
# time — each of those blocks on a 5 s httpx timeout, which would stretch
# the arbiter's cadence from 2 s to ~12 s exactly when nothing is wrong. So a
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
        (``observe_spark(None, node_id)`` then emits no key — an undeclared
        resource must not appear as a phantom failure)."""
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
                # Exponent capped: 2**(failures-1) is a Python bignum and
                # overflows float multiplication at failure #1025 (2**1024
                # is the first power of two too large for a float; ~3.5
                # days of a powered-off swap node at the 300 s cap). 2**16
                # already exceeds every real backoff_max_s.
                self._backoff_base_s * 2 ** min(self._failures - 1, 16),
                self._backoff_max_s,
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
