"""
Model Deck world-state snapshot.

``World.snapshot()`` assembles one point-in-time view of all tenants
(lemonade, comfyui, hipfire) plus GPU VRAM totals and a best-effort
"externals" list, from already-fetched engine/GPU data. This module does
no I/O of its own — the caller owns sysfs reads (``app.gpu.read_gpus``)
and every engine HTTP call, and passes the results/clients in. That keeps
the aggregation logic here pure and cheap to test, following the repo's
functional-core/imperative-shell convention.

A ``World`` instance holds two pieces of in-memory state across calls: the
last-seen lemonade activity-counter value/timestamp, and comfyui's
last-busy timestamp. Both are driven by an injectable ``clock`` (defaults
to ``time.monotonic``) so tests can move time forward deterministically
without real sleeps — a fresh ``World()`` per process is expected to be
re-created at most once (the arbiter/watcher owns exactly one instance
for its whole lifetime); idle clocks reset to "just started" if the
process restarts.

Per-engine failures: ``EngineError`` raised by an engine call during a
snapshot degrades only that tenant to ``state="unknown"`` (fields that
have no sensible zero value go to ``None``; hipfire's ``footprint`` is
typed as a plain ``int`` in the snapshot shape, not ``int | None``, so it
goes to ``0`` instead) rather than failing the whole snapshot. This is the
one exception type this module catches. Any other exception (e.g. a bare
``KeyError`` from a malformed response body) is a real bug and is allowed
to propagate per the house "let it crash" policy — the watcher loop that
calls ``snapshot()`` on a timer is expected to catch per-tick, not this
module.

hipfire's ``model`` name and ``default_route`` both come from
``litellm.route_table()``. A litellm failure there does NOT downgrade
hipfire's ``state`` to "unknown" — hipfire's operational state came back
fine from ``HipfireClient.status()`` independently, so only the ``model``
name (which litellm alone can supply) and ``default_route`` go to
``None``. Only a failure of hipfire's *own* status call marks the hipfire
tenant "unknown".

Externals (minimal heuristic, display-only — see task brief for the full
version this deliberately simplifies): a KFD pid using more than 1 GiB
counts as "external" only when NO tenant anywhere reports a loaded/running
state (lemonade "loaded", comfyui "busy", or hipfire "running"). This
heuristic doesn't scope to "the GPU that tenant is actually on" (that would
need attributing ``app.gpu.read_gpus`` pids to specific tenants, which is
explicitly out of scope here) even though ``placement`` now carries the
tenant->GPU index mapping. Practical effect: as soon as any tenant is
loaded/running anywhere in the box, all fat pids on all GPUs are treated as
accounted-for, even ones on a GPU with no tenant activity at all — this
under-reports externals whenever a tenant is loaded on one GPU while
unrelated heavy VRAM use happens on another. Acceptable for a display-only
UI list; do not use this for eviction decisions.

No Settings import here — pure inputs only.
"""

import time

from app.engines import EngineError
from app.registry import HIPFIRE_FOOTPRINT

# Externals heuristic floor: below this, transient/small allocations are
# noise, not worth surfacing as a tenant-shaped list entry.
_EXTERNAL_FLOOR_BYTES = 1 * 1024**3  # 1 GiB

_OPENAI_PREFIX = "openai/"
_EXTRA_PREFIX = "extra."

# Mirrors Settings' fixed engine->GPU placement defaults; _build_deck passes
# the live Settings values so the UI never has to hardcode the layout.
_DEFAULT_PLACEMENT = {"hipfire": 0, "lemonade": 1, "comfyui": 1}


def _strip_prefix(name: str | None, prefix: str) -> str | None:
    if name is None:
        return None
    return name.removeprefix(prefix)


class World:
    """In-memory idle-clock state across repeated ``snapshot()`` calls."""

    def __init__(self, clock=time.monotonic, placement: dict[str, int] | None = None) -> None:
        self._clock = clock
        self._lemonade_last_value: int | None = None
        self._lemonade_last_activity_time: float | None = None
        self._lemonade_last_loaded: str | None = None
        self._comfy_last_activity_time: float | None = None
        # Static tenant->GPU index mapping, carried explicitly from Settings
        # (see _DEFAULT_PLACEMENT / app.main._build_deck) — not inferred from
        # any engine call, so it never fails and never varies across a
        # process's lifetime.
        self._placement = dict(placement) if placement is not None else dict(_DEFAULT_PLACEMENT)

    def snapshot(self, gpus, lemonade, comfy, hipfire, litellm, registry) -> dict:
        now = self._clock()

        gpu_list = [
            {
                "index": gpu["index"],
                "total": gpu["vram_total"],
                "used": gpu["vram_used"],
                "free": gpu["vram_total"] - gpu["vram_used"],
            }
            for gpu in gpus
        ]

        try:
            routes = litellm.route_table()
        except EngineError:
            routes = None

        lemonade_tenant = self._snapshot_lemonade(lemonade, registry, now)
        comfy_tenant = self._snapshot_comfy(comfy, now)
        hipfire_tenant = self._snapshot_hipfire(hipfire, routes)
        default_route = None if routes is None else _strip_prefix(routes.get("default"), _OPENAI_PREFIX)

        externals = self._externals(gpus, lemonade_tenant, comfy_tenant, hipfire_tenant)

        return {
            "gpus": gpu_list,
            "tenants": {
                "lemonade": lemonade_tenant,
                "comfyui": comfy_tenant,
                "hipfire": hipfire_tenant,
            },
            "externals": externals,
            "default_route": default_route,
            # Disambiguates default_route=None: "litellm says there is no
            # default route" (True) vs "we could not reach litellm to ask"
            # (False). The storage guards fail CLOSED on the latter — see
            # app.storage.plan_move / storage_decide.
            "routes_known": routes is not None,
            "placement": dict(self._placement),
        }

    def _snapshot_lemonade(self, lemonade, registry, now: float) -> dict:
        # A deck-authored load in flight: llama-server health reports nothing
        # loaded while weights stream in, which upstream must not read as dead
        # (derive_status turns 'loading' into the inert 'warming'). Idle-clock
        # bookkeeping is deliberately skipped: the loaded-value transition on
        # the first post-load snapshot resets the activity clock as usual.
        if lemonade.load_in_flight():
            return {"state": "loading", "model": None, "footprint": None, "idle_s": None}

        try:
            status = lemonade.status()
        except EngineError:
            return {"state": "unknown", "model": None, "footprint": None, "idle_s": None}

        loaded = status["loaded"]
        activity = lemonade.activity()  # never raises

        # A load transition is activity: llama.cpp's counters restart at 0 on
        # every load, so a fresh model can report the same counter value the
        # previous one ended with and the value-change check alone would let
        # it inherit a stale idle clock (and be evicted on the next tick).
        if loaded != self._lemonade_last_loaded:
            self._lemonade_last_activity_time = now
        self._lemonade_last_loaded = loaded

        if activity is not None:
            if self._lemonade_last_value is None or activity != self._lemonade_last_value:
                self._lemonade_last_activity_time = now
            self._lemonade_last_value = activity
            idle_s = now - self._lemonade_last_activity_time
        else:
            idle_s = None

        footprint = None
        if loaded:
            key = _strip_prefix(loaded, _EXTRA_PREFIX)
            try:
                footprint = registry.footprint(key)
            except FileNotFoundError:
                footprint = None

        return {
            "state": "loaded" if loaded else "unloaded",
            "model": loaded,
            "footprint": footprint,
            "idle_s": idle_s,
        }

    def note_comfy_freed(self) -> None:
        """A successful comfy VRAM free re-arms the idle TTL. Without this,
        idle_s only grows once comfy is idle (freeing changes none of the
        idle-release rule's inputs), so the watcher re-emits free_comfyui on
        every tick — flooding the event ring and comfy's /free endpoint."""
        self._comfy_last_activity_time = self._clock()

    def _snapshot_comfy(self, comfy, now: float) -> dict:
        try:
            queue = comfy.queue_len()
        except EngineError:
            return {"state": "unknown", "queue": None, "idle_s": None}

        if self._comfy_last_activity_time is None:
            # First-ever snapshot: establish a baseline so idle_s is always
            # computable from here on, even if the queue happens to be
            # empty on this very first call.
            self._comfy_last_activity_time = now

        if queue > 0:
            self._comfy_last_activity_time = now
            state = "busy"
        else:
            state = "idle"

        idle_s = now - self._comfy_last_activity_time
        return {"state": state, "queue": queue, "idle_s": idle_s}

    def _snapshot_hipfire(self, hipfire, routes: dict | None) -> dict:
        try:
            state = hipfire.status()
        except EngineError:
            return {"state": "unknown", "model": None, "footprint": 0, "queue_depth": None}

        # Poll /stats while running: besides surfacing queue_depth, this is
        # what feeds the HipfireClient conversation-activity tracker every
        # watcher tick (the park/apply busy guard reads that tracker). A
        # stats failure must not take down the snapshot — unknown, not fatal.
        queue_depth = None
        if state == "running":
            try:
                queue_depth = hipfire.stats().get("queue_depth")
            except EngineError:
                queue_depth = None

        model = None if routes is None else _strip_prefix(routes.get("hipfire"), _OPENAI_PREFIX)
        footprint = HIPFIRE_FOOTPRINT if state == "running" else 0
        return {"state": state, "model": model, "footprint": footprint, "queue_depth": queue_depth}

    def _externals(self, gpus, lemonade_tenant, comfy_tenant, hipfire_tenant) -> list[dict]:
        any_tenant_active = (
            lemonade_tenant["state"] == "loaded"
            or comfy_tenant["state"] == "busy"
            or hipfire_tenant["state"] == "running"
        )
        if any_tenant_active:
            return []

        externals = []
        for gpu in gpus:
            for pid, nbytes in gpu["pids"].items():
                if nbytes > _EXTERNAL_FLOOR_BYTES:
                    externals.append({"pid": pid, "gpu": gpu["index"], "bytes": nbytes})
        return externals
