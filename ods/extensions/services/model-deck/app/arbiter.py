"""
Model Deck priority arbiter — the core decision engine.

Two pieces:

* ``decide(world, policy, pending_load) -> list[dict]`` — a PURE function
  (no I/O, deterministic) that turns one world snapshot plus the tenant
  policy plus an optional inferred pending load into a list of action dicts.
  This is where every eviction guard lives, so it is exhaustively unit
  tested in isolation without any thread or client.

* ``class Watcher`` — the imperative shell: a daemon thread that, every
  ``settings.watch_interval`` seconds, reads GPUs -> snapshots the world ->
  infers a pending load -> calls ``decide`` -> executes the returned actions
  via the engine clients -> logs each executed action. It owns exactly one
  ``World`` instance for its whole lifetime (idle clocks live there).

Action dicts (the only three shapes ``decide`` ever returns; E1 generalizes
every action to carry the RESOURCE it targets, not a kind-baked verb —
``unload_lemonade``/``free_comfyui`` are gone):
    {"type": "unload", "resource": <str>, "model": <str>}
    {"type": "free", "resource": <str>}
    {"type": "noop", "reason": <str>}   # reason in {"fits", "wont-fit"}

Design notes that are load-bearing (and tested):

* IDLE RELEASE (rule 1) is allowed on a demand-routing kind's (lemonade)
  default-route model. That is the deliberate idle-GPU-burn fix — a
  resident-but-idle llama.cpp model pins a card at 100%/86 W doing nothing;
  unloading drops it to ~17 W and the next request reloads it in ~4 s. This
  exception applies ONLY to idle release, never to contention. E1: this rule
  now runs independently for EVERY declared resource (not a single hardcoded
  "lemonade" tenant) via each kind's own ``idle_action`` (app.engine_kinds).

* CONTENTION HEALING (rule 2) NEVER evicts a demand-routing kind's
  default-route model, no matter how much VRAM that would free. It also
  never touches a kind with no arbiter verb (hipfire has none — park stays
  human-only) and never frees a busy or unknown-state comfy queue.
  Candidates are scoped to the PENDING LOAD'S OWN GPU (``tenant["gpu_index"]
  == gpu["index"]``) — E1 generalizes past the old single-GPU assumption, so
  two contended GPUs at once no longer cross-contaminate each other's
  candidate sets — then sorted ascending by ``policy[resource]["priority"]``
  (lowest priority evicted first) — with the default policies (comfyui 40,
  lemonade 50) this reproduces the historical comfyui-then-lemonade order.

* Feasibility-first eviction: if the FULL set of eligible evictions still
  can't free enough VRAM, ``decide`` emits ``noop "wont-fit"`` and evicts
  NOTHING. We never kill a tenant for a load that won't fit anyway.

* HEAL SUPPRESSION (``HealSuppressor``): contention healing infers a pending
  default-route load whenever the route is set, lemonade is unloaded, and
  free VRAM < footprint. Left unchecked that inference auto-REVERTS any
  deliberate unload — a manual unload, a set-apply ``unload_lemonade`` step,
  or even the watcher's own idle release (the idle-GPU-burn fix). To stop
  that, EVERY deck-initiated lemonade unload arms a shared, monotonic-clock
  suppression window (``note_deck_unload()``); while it's active the watcher
  skips pending-load inference entirely (idle rules still run). Any
  deck-initiated lemonade LOAD (manual, set-apply, or the watcher's own heal
  re-trigger) means the model is wanted resident again, so it clears the
  suppression. Accepted trade-off: during the window a chat request against a
  full GPU can still fail Lemonade-side (no auto-heal); healing resumes once
  the window expires. A future refinement would replace the timer with a real
  load-failure signal (e.g. tailing the llama-server log via the docker
  proxy) so healing can distinguish "user unloaded on purpose" from "a load
  actually OOM'd".

* The watcher YIELDS to an in-flight actuator (task 6, ``app.actuation``): the
  snapshot and ``decide()`` still run every tick (cheap, read-only), but the
  actuation+reconcile phase TRY-acquires ``app.actuation.LOCK`` and skips
  cleanly — no engine calls, no restores — whenever a set apply or the
  pull-through completion hook already holds it. This replaced a start-of-
  tick PEEK against ``app.sets.apply_in_progress()``: a tick already past
  that peek kept actuating even after an apply had since started, so its
  restores could interleave with the apply's own evictions/loads
  [max-review #7]. Gating at the actuation boundary instead of at tick start
  closes that window structurally — there is no point after the try-acquire
  where "someone else is now actuating" can go unnoticed.

* The watcher also YIELDS to an in-flight host-agent lifecycle operation
  (activation/download/delete): that machinery owns the box's model state
  end-to-end (snapshots, container restarts, readiness proofs), and a deck
  action fired mid-operation races it — e.g. an unload can make the agent's
  readiness gate fail and roll back for no reason. When there is real work
  to do (a pending load or a non-noop action), the watcher probes
  ``hostagent.lifecycle()``; if it reports active, the whole tick is skipped
  and a deduped ``host-agent-busy`` event is logged instead. An idle tick
  (nothing to do) never probes the agent at all. ``hostagent=None`` (the
  default) disables the check entirely, so this is opt-in.

* LIFECYCLE RECONCILIATION (``_reconcile_pass``) is a second, independent
  pass at the END of every tick. Arbitration answers "who must give up VRAM
  right now"; reconciliation answers "is what the operator asked for still
  true", by comparing app.intent against this tick's observation. It is what
  stops ods-hipfire sitting Exited(0) for 26 hours looking exactly like a
  deliberate park (2026-08-03). ``intent_store=None`` (the default) disables
  it entirely — same opt-in shape as ``hostagent``.

* CHARACTERISTICS DERIVE PASS (``_derive_pass``) is a third, independent pass
  at the very end of every tick, refreshing app.characteristics from local
  checkpoints (``gguf_dir``) and live engine surfaces (``spark``). Throttled
  to ``settings.derive_interval_s`` (default 300 s) because the tick loop
  runs every couple of seconds and checkpoint directories do not change that
  fast — EXCEPT the first successful ``_reconcile_pass`` restore of an
  incident clears the throttle (``self._last_derive_at = None`` in
  ``_execute_restore``, no new I/O in the restore path itself), so a resource
  that just came back up gets its live facts captured by that same tick's
  derive pass instead of waiting up to ``derive_interval_s``. That clear is
  itself floor-limited (``_DERIVE_RESTORE_FLOOR_S``, 30 s): a crash-looping
  resource can have ``_restore()`` succeed at the API level without ever
  raising; the failure budget charges that case on the next dispatch for a
  still-unverified key (``_restore_unverified``), but quarantine only lands
  after FAILURE_BUDGET cooldown-paced charges — without the floor every
  restore in that window would re-clear the throttle and re-run the scan.
  ``characteristics_store=None`` (the default) disables the whole pass —
  same opt-in shape as ``hostagent``/``intent_store``.
"""

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from app import actuation, provenance_collect
from app.derive_checkpoint import derive_checkpoint
from app.derive_live import derive_live_models
from app.engine_kinds import ENGINE_KINDS
from app.engines import BusyError, EngineError, GuardError
from app.events import log_event
from app.harvest import PROBE_INTERPRETER, PROBE_SOURCE, parse_probe_output
from app.intent import FAILURE_BUDGET
from app.lifecycle import derive_status
from app.observe import (
    _LOCAL_NODE,
    LOCAL_LEMONADE_KEY,
    merge_observations,
    observe_local,
    observe_spark,
)
from app.reconcile import plan_reconcile

_EXTRA_PREFIX = "extra."

# Minimum spacing between restore-triggered characteristics-derive-throttle
# clears (see Watcher._execute_restore). A crash-looping resource can have
# _restore() succeed at the API level (e.g. resume() returns fine) while the
# process dies again before the next tick — never raising. The failure
# budget now bounds that case too (a cooldown-paced redispatch of a
# still-unverified key charges it, see _restore_unverified), but quarantine
# only lands after FAILURE_BUDGET charges (~60-90 s of flapping); without a
# floor, every restore in that window would clear the derive throttle and
# re-run the checkpoint scan + spark probe, exactly the pointless I/O the
# throttle exists to prevent, concentrated during an active incident. 30 s
# bounds that to at most one extra derive per half-minute of crash-looping —
# well above the ~2 s tick so one flap cycle can't retrigger it, and well
# below settings.derive_interval_s's default 300 s so a genuinely new
# incident more than 30 s later still gets its own immediate derive.
_DERIVE_RESTORE_FLOOR_S = 30.0

# Minimum spacing between restore attempts FOR THE SAME KEY (storm limiter,
# defense-in-depth for observation windows the earlier tasks don't cover —
# deck-invisible lazy loads, hipfire/spark record-last windows). A restore
# that "succeeds" at the API level and dies out-of-band before the next tick
# never raises; without this floor, reconcile re-dispatches a real restore
# every ~2 s tick for as long as the resource keeps flapping (observed live:
# three restores in 10 s, 2026-08-06). The failure budget
# (app.intent.FAILURE_BUDGET) charges that no-raise case on the NEXT
# dispatch for a still-unverified key (see _restore_unverified), so this
# cooldown is also what paces those charges. A skipped restore is a
# non-action: no event, no failure charged.
_RESTORE_COOLDOWN_S = 30.0

# Engine failures carry raw HTTP bodies (`EngineError(resp.text)`), and a
# multi-KB body re-logged per 2 s tick is exactly the trim-thrash input
# app/events.py's byte-denominated trim bounds [T10]. Bound the DETAIL too:
# 500 chars keeps a failed-load line well under events' 2500 B/line
# hysteresis ratio even when the body is unstable tick to tick (timestamps
# in an error page defeat dedup, so truncation is the backstop dedup needs).
_ERROR_TEXT_MAX = 500


def _error_text(exc: BaseException) -> str:
    text = str(exc)
    if len(text) <= _ERROR_TEXT_MAX:
        return text
    return text[:_ERROR_TEXT_MAX] + " …[truncated]"


# ===========================================================================
# Heal suppression — shared flag guarding deliberate unloads
# ===========================================================================


class HealSuppressor:
    """A shared, monotonic-clock suppression window that stops the watcher's
    contention healing from auto-reverting a deliberate lemonade unload.

    ``note_deck_unload()`` (arm) is called by EVERY deck-initiated lemonade
    unload — watcher idle-release and contention evictions, the set-apply
    ``unload_lemonade`` step, and the manual unload route. While
    ``suppressed()`` is True the watcher skips pending-load inference entirely.

    Any deck-initiated lemonade LOAD calls ``clear()`` — the model is wanted
    resident again, so there is nothing to protect from re-loading.

    The clock is injectable so tests can advance the window without sleeping.
    """

    def __init__(self, window_s: float, clock=time.monotonic) -> None:
        self._window_s = window_s
        self._clock = clock
        self._until: float | None = None

    def note_deck_unload(self) -> None:
        """Arm (or re-arm) the suppression window from now."""
        self._until = self._clock() + self._window_s

    def clear(self) -> None:
        """Disarm the window (a deck-initiated load wants the model resident)."""
        self._until = None

    def suppressed(self) -> bool:
        """True while the window is active. Lazily disarms once it expires."""
        if self._until is None:
            return False
        if self._clock() >= self._until:
            self._until = None
            return False
        return True


# ===========================================================================
# Pure decision function
# ===========================================================================


def decide(world: dict, policy: dict, pending_load: dict | None) -> list[dict]:
    """Pure arbitration: world snapshot + policy + optional pending load ->
    ordered list of action dicts. No I/O, no engine calls.

    A pending load (contention) takes precedence over idle release: when a
    load is waiting for VRAM we heal the contention rather than idly release.
    """
    if pending_load is not None:
        return _decide_contention(world, policy, pending_load)
    return _decide_idle_release(world, policy)


def _decide_idle_release(world: dict, policy: dict) -> list[dict]:
    """Rule 1, E1-generalized: every DECLARED resource gets its own idle
    check, not a hardcoded lemonade+comfyui pair. Iterates
    ``world["tenants"]`` sorted by resource (deterministic order) and asks
    that resource's kind adapter (app.engine_kinds) whether it wants to
    idle-release — the two rules' actual guard conditions moved there
    VERBATIM, comments included (they carry incident history)."""
    actions: list[dict] = []
    tenants = world["tenants"]
    gpus = world["gpus"]

    for resource in sorted(tenants):
        tenant = tenants[resource]
        # Tolerate a policy map missing this resource (partial/legacy
        # declaration, or a stale test fixture): no policy row means there
        # is nothing safe to decide for it, never a KeyError.
        pol = policy.get(resource)
        if pol is None:
            continue
        adapter = ENGINE_KINDS[tenant["engine"]]
        gpu_index = tenant.get("gpu_index")
        gpu = _find_gpu(gpus, gpu_index) if gpu_index is not None else None
        co_footprints = _co_resident_footprints(tenants, resource, gpu_index)
        action = adapter.idle_action(tenant, pol, gpu, co_footprints)
        if action is not None:
            actions.append({**action, "resource": resource})

    return actions


def _co_resident_footprints(tenants: dict, exclude_resource: str, gpu_index) -> int:
    """Sum of known footprints of every OTHER tenant sharing `gpu_index`
    with a loaded/running state — the generalization of the old "minus a
    co-resident loaded lemonade's footprint": a kind's own VRAM use may not
    be directly observable (comfyui), so its reclaimable estimate must
    subtract whatever ANY OTHER co-resident tenant is already accounted
    for, not a hardcoded single-kind check."""
    total = 0
    for resource, tenant in tenants.items():
        if resource == exclude_resource:
            continue
        if tenant.get("gpu_index") != gpu_index:
            continue
        if tenant.get("state") not in ("loaded", "running"):
            continue
        footprint = tenant.get("footprint")
        if isinstance(footprint, int) and footprint:
            total += footprint
    return total


def _decide_contention(world: dict, policy: dict, pending_load: dict) -> list[dict]:
    footprint = pending_load["footprint"]
    gpu = _find_gpu(world["gpus"], pending_load["gpu_index"])
    if gpu is None:
        return [_noop("wont-fit")]

    free = gpu["free"]
    if free >= footprint:
        return [_noop("fits")]

    candidates = _eviction_candidates(world, policy, gpu)

    # Feasibility first: only evict when the eligible set actually resolves
    # the contention. Never kill a tenant for a load that won't fit.
    total_reclaimable = sum(reclaimable for _, reclaimable in candidates)
    if free + total_reclaimable < footprint:
        return [_noop("wont-fit")]

    # Feasible — emit the minimal ascending-priority prefix that fits.
    actions: list[dict] = []
    projected = free
    for action, reclaimable in candidates:
        actions.append(action)
        projected += reclaimable
        if projected >= footprint:
            break
    return actions


def _eviction_candidates(world: dict, policy: dict, gpu: dict) -> list[tuple[dict, int]]:
    """Evictable tenants on the pending GPU (``tenant["gpu_index"] ==
    gpu["index"]`` — E1: candidates are scoped to the GPU the pending load
    actually targets, not every declared resource) as ``(action,
    reclaimable_bytes)``, sorted ascending by ``policy[resource]["priority"]``
    (lowest priority evicted first). Eligibility guards are unchanged from
    before; only the resulting order is policy-driven.

    A kind with no arbiter verb (hipfire) is never a candidate — the guard
    against touching it is by structural omission here.
    """
    tenants = world["tenants"]
    default_route = world.get("default_route")

    scored: list[tuple[int, dict, int]] = []
    for resource in sorted(tenants):
        tenant = tenants[resource]
        if tenant.get("gpu_index") != gpu["index"]:
            continue
        adapter = ENGINE_KINDS[tenant["engine"]]
        if not adapter.arbiter_verbs():
            continue  # e.g. hipfire: structurally excluded
        pol = policy.get(resource)
        if pol is None or pol["pinned"]:
            continue
        # ABSOLUTE guard, rule 2 ONLY (never rule 1 — see idle_action's own
        # NOTE): a demand-routing kind's currently-loaded default-route
        # model must never be evicted for contention, no matter how much
        # VRAM that would free.
        if (
            adapter.demand()
            and tenant.get("model") is not None
            and tenant.get("model") == default_route
        ):
            continue
        co_footprints = _co_resident_footprints(tenants, resource, gpu["index"])
        reclaimable = adapter.reclaimable(tenant, gpu, co_footprints)
        if reclaimable is None:
            continue  # unquantifiable (or, for a "free"-verb kind, busy/unknown)

        # Build the action from the kind's single arbiter verb — "model" is
        # present in `tenant` iff the verb needs it (lemonade's "unload";
        # comfyui's obs never carries a "model" key at all), so this reads
        # the payload shape off the tenant itself rather than branching on
        # a kind name (spec §8: no engine name outside app.engine_kinds).
        (verb,) = adapter.arbiter_verbs()
        action = {"type": verb, "resource": resource}
        if "model" in tenant:
            action["model"] = tenant["model"]

        scored.append((pol["priority"], action, reclaimable))

    # Stable sort: ties keep the sorted-by-resource insertion order above.
    scored.sort(key=lambda c: c[0])
    return [(action, reclaimable) for _, action, reclaimable in scored]


def _find_gpu(gpus: list[dict], gpu_index: int) -> dict | None:
    return next((g for g in gpus if g["index"] == gpu_index), None)


def _noop(reason: str) -> dict:
    return {"type": "noop", "reason": reason}


# ===========================================================================
# Catalog harvest — module-level so both Watcher._harvest_catalogs (the
# per-tick loop, force=False) and the manual force-harvest HTTP route
# (app.routers.settings.harvest_now, force=True) call the exact same logic.
# ===========================================================================


def harvest_catalog_pair(engine_exec, characteristics_store, log, node, engine,
                          now, force=False) -> dict:
    """Harvest one (node, engine) pair's option catalog, once per version.

    Costs a docker exec and a vLLM import, so calling `engine_exec` AT ALL
    must be avoidable once a version is already cached and nothing has
    changed -- comparing versions AFTER paying for the exec would defeat
    the whole point (every derive pass would still pay the cost it exists
    to avoid). `engine_exec` may expose the engine's currently observed
    version as a plain `.version` attribute -- a cheap peek (a `docker
    inspect`, not an exec, in the real DockerCtl-backed implementation)
    read WITHOUT invoking it -- and when that peek is available and
    matches what's cached, the exec is skipped entirely. `engine_exec`
    without a `.version` attribute (or any other caller) still works
    correctly, just without the early-skip optimization: the post-call
    version comparison below is the fallback, so a redundant exec never
    produces a redundant write. `engine_version` (both the peek and the
    post-call value) is an OPAQUE identity used only for change detection
    -- the real DockerCtl-backed adapter feeds it the container's resolved
    image content ID, not a human-readable engine version string; a UI or
    log line rendering it as "the vLLM version" would be wrong.

    `force=True` (the manual force-harvest route) skips BOTH version gates
    -- the cheap peek and the post-call compare -- so a human pressing
    Refresh always pays for a real exec and always gets a fresh write,
    never "current" for a peek that only LOOKS unchanged.

    An engine that is not running (EngineError/BusyError) or whose
    container isn't in the deck's own park allowlist (GuardError --
    deliberately not an EngineError subclass, see app.engines) simply
    yields no catalog -- a supported state, since validation warns rather
    than blocks (app.validate_settings). It still logs a deduped
    ``harvest-failed`` event (F3, final branch review, 2026-08-07): "no
    catalog" alone can't tell "engine parked" apart from a real, persistent
    exec failure (e.g. the proxy's ``-allowPOST`` lines missing from a
    hand-merged live compose -- see the branch's own recorded deploy
    hazard), and the two need to be distinguishable from the outside. This
    function must never raise past this point: both callers (the watcher's
    tick loop and the manual HTTP route) need a foreseeable-failure result,
    not an exception.
    """
    key = f"engine/{node}/{engine}"
    cached = characteristics_store.entry(key).get("option_catalog")
    cached_version = cached["value"].get("engine_version") if cached else None

    if not force:
        peeked_version = getattr(engine_exec, "version", None)
        if cached_version is not None and peeked_version == cached_version:
            return {"outcome": "current"}

    try:
        version, output = engine_exec(node, engine, PROBE_INTERPRETER, PROBE_SOURCE)
    except (EngineError, BusyError, GuardError) as exc:
        # F3, Important (final branch review, 2026-08-07): this branch used
        # to emit nothing at all, which makes it indistinguishable from
        # "engine parked" forever -- and the branch's own recorded deploy
        # hazard (a live compose hand-merge losing the proxy's -allowPOST
        # lines -> 403 on exec) lands EXACTLY here. Deduped like
        # harvest-empty (by the caller's own _log, for Watcher; log_event
        # is not deduped for the manual route -- see harvest_now). The
        # detail is the exception's CLASS only, not its message text: a
        # real failure's message (a proxy's response body, a transport
        # error string) is not guaranteed stable call to call, and an
        # unstable detail would defeat the dedup entirely -- re-logging
        # every derive_interval_s is the exact spam this exists to avoid.
        log("harvest-failed", {"key": key, "reason": type(exc).__name__})
        return {"outcome": "failed", "reason": type(exc).__name__}

    if not force and cached_version is not None and cached_version == version:
        return {"outcome": "current"}

    field = parse_probe_output(output, engine_version=version, now=now)
    if not field:
        # The exec ran (engine reachable, allowlisted) but produced no
        # parseable catalog -- e.g. a probe crash inside the container.
        # Without this, "never harvested" and "harvest is broken" look
        # identical from the outside: no catalog, no error, forever.
        # Deduped (_DEDUP_KINDS) on the Watcher path -- a persistently
        # broken probe would otherwise re-log every derive_interval_s --
        # but NOT on the manual route, whose log_event is undeduped by
        # design (see harvest_now): a human pressing Refresh twice wants
        # two events, not the second one silently swallowed.
        log("harvest-empty", {"key": key})
        return {"outcome": "empty"}

    characteristics_store.put_fields(key, {"option_catalog": field})
    log("catalog-harvested", {"key": key, "engine_version": version})
    return {"outcome": "harvested", "options": len(field["value"]["options"])}


# ===========================================================================
# Watcher — imperative shell (daemon thread)
# ===========================================================================


class Watcher:
    """Periodic arbiter loop. Owns one ``World`` and drives ``decide`` on a
    timer, executing actions against the engine clients and logging each.

    All deps are injected (including ``read_gpus``) so tests need no thread to
    cover behavior; one thread test covers start()/stop() liveness.
    """

    def __init__(
        self,
        settings,
        world,
        lemonade,
        comfy,
        hipfire,
        litellm,
        registry,
        policy_store,
        events_path,
        read_gpus,
        heal_suppressor=None,
        hostagent=None,
        catalog=None,
        intent_store=None,
        node_clients=None,
        node_observers=None,
        characteristics_store=None,
        gguf_dir=None,
        clock=time.monotonic,
        on_derive=None,
        engine_exec=None,
        configurable_engines=None,
        provenance_store=None,
        dockerctl=None,
        node_store=None,
        local_clients=None,
    ) -> None:
        self._settings = settings
        self._world = world
        self._lemonade = lemonade
        self._comfy = comfy
        self._hipfire = hipfire
        self._litellm = litellm
        self._registry = registry
        self._policy_store = policy_store
        self._events_path = events_path
        self._read_gpus = read_gpus
        # Declaration + LocalClients (E1 Task 3): what the tick's
        # World.snapshot call reads through now. Both None (every unit test
        # except the ones built through app.main._build_watcher) means "no
        # declared engines" — the tick's own snapshot degrades to an empty
        # `engines` list, same as a genuinely fresh install with nothing
        # declared yet (spec §1, seed_engines_if_missing's docstring).
        # COEXISTENCE: observation only — self._lemonade/_comfy/_hipfire
        # above still drive every ACTUATION path in this class unchanged.
        self._node_store = node_store
        self._local_clients = local_clients
        # Shared across the HTTP routers (set-apply, manual load/unload) via
        # the deck namespace; a standalone default keeps unit tests simple.
        self._heal_suppressor = (
            heal_suppressor
            if heal_suppressor is not None
            else HealSuppressor(settings.heal_suppress_s)
        )
        self._hostagent = hostagent
        # Optional storage catalog (app.catalog.Catalog): a successful heal
        # re-trigger is a real load, and the storage watcher's LRU eviction
        # order is only as good as the loads it gets told about. None (unit
        # tests, any caller without the deck) simply skips the bookkeeping.
        self._catalog = catalog
        # Durable desired state, shared with the HTTP routers (which write it
        # on every deliberate action). None disables the reconcile pass, which
        # is what every pre-lifecycle caller and unit test gets.
        self._intent_store = intent_store
        # Per-node actuation clients + observation caches (app.node_clients).
        # None (unit tests, pre-N1 callers) means no swap nodes: observation
        # emits no slot keys and the spark restore branch refuses.
        self._node_clients = node_clients
        self._node_observers = node_observers
        self._interval = settings.watch_interval

        # Characteristics derive pass: refreshes app.characteristics from
        # local checkpoints (gguf_dir) and live engine surfaces (spark),
        # throttled to settings.derive_interval_s. characteristics_store=None
        # (the default) disables it entirely — same opt-in shape as
        # hostagent/intent_store above.
        self._characteristics_store = characteristics_store
        self._gguf_dir = gguf_dir
        self._clock = clock
        self._derive_interval_s = settings.derive_interval_s
        self._last_derive_at: float | None = None
        # Floor between restore-triggered throttle clears (see
        # _execute_restore) — separate from _last_derive_at so it tracks only
        # restores, not the regular timed derive passes.
        self._last_restore_derive_at: float | None = None
        # Per-key restore cooldown (storm limiter, see _RESTORE_COOLDOWN_S):
        # last monotonic time a restore was ATTEMPTED for this key, success
        # or failure. Only intent-store callers ever populate it.
        self._restore_last_attempt_at: dict[str, float] = {}
        # Keys whose last restore dispatched fine (no raise) but has not yet
        # produced a `serving` tick. The reconciler being about to dispatch
        # the NEXT restore for a still-marked key IS the evidence the
        # previous restore failed, so that dispatch charges note_failure
        # first (see _reconcile_pass) — closing the no-raise hole in the
        # failure budget (live-proven 08-12: unbounded 20-min restore loop,
        # no quarantine). Deliberately in-memory, same lifetime as
        # _restore_last_attempt_at: a deck restart forgets at most ONE
        # pending charge, and the incident class involves a long-running
        # deck; persisting it would touch the intent-store schema for no
        # incident-class gain.
        self._restore_unverified: set[str] = set()
        # Per-node memo for lifecycle-node-misconfigured (see
        # _node_observations): node id -> the warning text already logged.
        # Same posture as _last_failure_key (its own memo, not the one-slot
        # _last_event_key, which interleaved events defeat): log once per
        # (node, warning-text) incident, re-armed when the warning
        # disappears so fixed-then-broken-again logs a fresh event — the
        # dedup-key-never-cleared class the T9-fix review forbade.
        self._node_misconfig_logged: dict[str, str] = {}
        # Test-only seam: when set, called once per non-throttled derive pass
        # INSTEAD of the real checkpoint/engine scan, so the throttle timing
        # itself can be tested without a real gguf_dir or spark client.
        # Always None in production (app.main never passes it).
        self._on_derive = on_derive
        # Catalog harvest (see _harvest_catalogs): runs `(node, engine,
        # interpreter, source) -> (version, stdout)` inside the running
        # engine container. None disables harvest entirely — same opt-in
        # shape as characteristics_store/hostagent/intent_store above (and
        # every unit test except the harvest-scoped ones gets it).
        self._engine_exec = engine_exec
        # (node, engine) pairs to harvest an option catalog from, returned
        # by _configurable_engines VERBATIM (see its docstring) -- C2:
        # Watcher does no pairing of its own anymore. Production (app.main)
        # builds its one real pair with `(the node's registry id, "vllm")`,
        # never `settings.node_label`/`settings.spark_node_name` -- see
        # _configurable_engines' docstring for the historical bug that rule
        # guards against. None (every caller except the harvest-scoped unit
        # tests and app.main) leaves _configurable_engines returning [].
        self._configurable_engine_pairs = configurable_engines
        # Provenance ledger (app.provenance). None -- every unit test and
        # every pre-provenance caller -- disables the pass entirely, the same
        # opt-in shape as characteristics_store/intent_store/hostagent above.
        # `dockerctl` is the LOCAL oci source; each swap node's client
        # (fetched per node from self._node_clients, never held) is the
        # remote one; the catalog (already held) is weights. The pass
        # RECORDS ONLY -- see _provenance_pass.
        self._provenance_store = provenance_store
        self._dockerctl = dockerctl
        self._provenance_interval_s = settings.provenance_interval_s
        self._last_provenance_at: float | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_event_key = None
        self._last_failure_key = None

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="model-deck-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Interval-bounded join: the loop waits on the Event, so it wakes
            # immediately on stop; the timeout is a safety net, not the norm.
            self._thread.join(timeout=self._interval + 5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            if self._stop.wait(self._interval):
                break

    # --- one tick ----------------------------------------------------------

    def tick(self) -> None:
        # DELIBERATE broad catch: this is a supervisor loop. A crash in any
        # single tick (malformed engine body, transient client bug, a bad
        # snapshot) must NOT take the whole arbiter down — loop survival
        # trumps the house 'let it crash' default HERE, and only here. The
        # error is logged so it's never silent.
        try:
            gpus = self._read_gpus(self._settings.drm_root, self._settings.kfd_root)
            # Re-read the declaration fresh every tick (the NodeObservers
            # precedent: an edit applies live, no restart) rather than a
            # boot-time copy. `self._node_store is None` (every unit test
            # that doesn't go through app.main._build_watcher) means "no
            # declared engines" — an empty list, not a crash.
            engines = []
            if self._node_store is not None:
                local = self._node_store.get("local")
                engines = local.get("engines", []) if local is not None else []
                if self._local_clients is not None:
                    self._local_clients.retire_absent({e["resource"] for e in engines})
            world = self._world.snapshot(
                gpus, engines, self._local_clients, self._litellm, self._registry
            )
            policy = self._policy_store.get()
            # While a deliberate unload's suppression window is active, skip
            # pending-load inference entirely so healing can't revert it. Idle
            # rules (decide with pending=None) still run.
            pending = None if self._heal_suppressor.suppressed() else self._infer_pending(world)
            actions = decide(world, policy, pending)

            # A host-agent lifecycle operation (activation/download/delete)
            # owns the box's model state end-to-end — snapshots, container
            # restarts, readiness proofs. Deck actions fired mid-operation
            # race that machinery (e.g. an unload makes the agent's readiness
            # gate fail and roll back for no reason), so the watcher yields
            # whole ticks while one is active. Probed only when there is real
            # work to skip — an idle deck never polls the agent.
            real_work = pending is not None or any(a["type"] != "noop" for a in actions)
            if real_work and self._hostagent is not None:
                lifecycle = self._hostagent.lifecycle()
                if lifecycle["active"]:
                    self._log("host-agent-busy", {
                        "operation": lifecycle["operation"],
                        "target": lifecycle["target"],
                    })
                    return

            # The one process-wide actuation lock (app.actuation, task 6),
            # shared with set-apply and the pull-through completion hook.
            # TRY-acquire: someone else actuating real engine state right now
            # makes this tick a clean no-op from here on — same observable
            # yield as the old start-of-tick peek, minus its race [max-review
            # #7], since this check happens at the actual moment of
            # actuation rather than once before the snapshot. Re-plan from a
            # fresh snapshot next tick; nothing above this line touched
            # engine state, so there is nothing to undo.
            if not actuation.LOCK.acquire(blocking=False):
                return
            try:
                actuated_keys = self._execute(actions, pending)
                self._reconcile_pass(world, actuated_keys)
            finally:
                actuation.LOCK.release()
            self._derive_pass()
            self._provenance_pass()
        except Exception as exc:  # noqa: BLE001 — supervisor loop, see comment above
            self._log("tick-error", {"error": str(exc)})

    def _infer_pending(self, world: dict) -> dict | None:
        """Infer a pending default-route load waiting on VRAM.

        Only when: a default route is configured, lemonade currently has
        nothing loaded, the model is a GGUF we know how to size, and the
        lemonade GPU's free VRAM is below that footprint.
        """
        default_route = world["default_route"]
        if not default_route:
            return None
        if world["tenants"]["lemonade"]["state"] != "unloaded":
            return None

        key = default_route.removeprefix(_EXTRA_PREFIX)
        try:
            footprint = self._registry.footprint(key)
        except FileNotFoundError:
            return None  # not a loadable GGUF -> nothing to heal

        gpu_index = self._settings.lemonade_gpu_index
        gpu = _find_gpu(world["gpus"], gpu_index)
        if gpu is None or gpu["free"] >= footprint:
            return None

        return {"model": default_route, "footprint": footprint, "gpu_index": gpu_index}

    def _execute(self, actions: list[dict], pending: dict | None) -> set[str]:
        """Perform this tick's arbitration actions. Returns the intent-store
        keys actuated (unloaded or load-retriggered) THIS tick — the caller
        (tick()) feeds it to _reconcile_pass so a stale, pre-action snapshot
        can never restore-fight a change this same tick just made."""
        wont_fit = False
        eviction_raced = False
        actuated: set[str] = set()

        for action in actions:
            # E1 Task 5 generalized decide()'s own action vocabulary to
            # {"type": "unload"/"free"/"noop", "resource": r, ...} — this
            # mechanical translation back to the pre-E1 dispatch keys
            # ("unload_lemonade"/"free_comfyui") is the MINIMUM needed to
            # keep this method (and its event-log vocabulary) working
            # against the new action shapes without changing what it
            # actually does. Task 6 owns generalizing `_execute` itself
            # (and `_infer_pending`/`_restore`) past the single hardcoded
            # lemonade/comfyui pair — see this method's own docstring scope.
            kind_raw = action["type"]
            resource = action.get("resource")
            if kind_raw == "unload" and resource == "lemonade":
                kind = "unload_lemonade"
                # Whoever actuates, records — and records FIRST, so a tick
                # that lands mid-unload derives 'parked', never 'down'
                # (2026-08-06). actor="deck": this is the arbiter's OWN
                # automatic action (idle-release or contention-eviction),
                # never an operator's — app.routers.control's pull-through
                # supersession check relies on that distinction to tell
                # "the deck unloaded something idle" apart from "the
                # operator asked for this" [max-review Important-1].
                # Snapshot the prior record first, so a FAILED unload can roll
                # it back: the actuation didn't happen, and leaving 'unloaded'
                # standing against a still-loaded model derives the inert
                # 'unexpected' status until an eventual retry [max-review #9].
                prior = None
                if self._intent_store is not None:
                    prior = self._intent_store.get().get(LOCAL_LEMONADE_KEY)
                    self._intent_store.record(
                        LOCAL_LEMONADE_KEY, state="unloaded", model=None, engine="lemonade",
                        actor="deck")
                # Added BEFORE the engine call (mirrors the retrigger tail
                # below) so the invariant is structural: an unload wrapped in
                # try/except in the future still can't return without this
                # key marked actuated.
                actuated.add(LOCAL_LEMONADE_KEY)
                try:
                    self._lemonade.unload(action["model"])
                except EngineError as exc:
                    # Per-action isolation — _execute_restore's documented
                    # invariant (see its docstring), applied to the arm that
                    # lacked it: a raise here must not abort the remaining
                    # actions or the reconcile/derive/provenance passes via
                    # tick()'s broad catch [max-review #9].
                    #
                    # Logged BEFORE the rollback, not after: the diagnostic is
                    # what tells an operator this happened at all, so it must
                    # not be contingent on the rollback succeeding [T7 review].
                    self._log("unload-failed",
                              {"model": action["model"],
                               "error": _error_text(exc)})
                    # A real compare-and-swap: the predicate and the write
                    # run in ONE critical section inside the store
                    # (put_back_if). An operator can record a deliberate
                    # load/unload during the seconds this engine call hung,
                    # and blindly putting `prior` back would silently revert
                    # THEIR action — the same class of bug as the pull-through
                    # supersession hole task 6 closed. An earlier version read
                    # the record and then wrote in two separate critical
                    # sections, which is check-then-act, not CAS [T7/T8
                    # fix-round review].
                    #
                    # Still the speculative record this arm wrote? It is
                    # deck-authored AND unloaded; an operator's write flips
                    # actor to "operator" (their routes never pass
                    # actor="deck"), so that pair is a sufficient witness.
                    def _still_ours(current):
                        return (current is not None
                                and current.get("actor") == "deck"
                                and current.get("state") == "unloaded")

                    if self._intent_store is not None and prior is not None:
                        if not self._intent_store.put_back_if(
                                LOCAL_LEMONADE_KEY, _still_ours, prior):
                            # Someone else's intent is newer. Leave it alone
                            # and say so — a silent skip here would be the
                            # same invisibility the rollback exists to end.
                            # Covers the forget()-shaped case too (current
                            # gone entirely), which the previous shape logged
                            # nothing for.
                            self._log("unload-rollback-skipped",
                                      {"model": action["model"],
                                       "reason": "intent changed during the unload"})
                    # prior is None (no record for this key yet) + failure
                    # leaves the fresh 'unloaded' record standing: there is no
                    # forget() to undo it with. Rare, and no worse than the
                    # behavior this fix replaces.
                else:
                    # Same re-arm as the load arm's success path: a
                    # successful unload between two identical unload
                    # failures must not swallow the second one.
                    self._last_failure_key = None
                    # Deck-initiated unload (idle release OR contention
                    # eviction): arm suppression so healing can't immediately
                    # revert it.
                    self._heal_suppressor.note_deck_unload()
                    self._log(kind, {"model": action["model"]})
            elif kind_raw == "free" and resource == "comfyui":
                kind = "free_comfyui"
                try:
                    self._comfy.free()
                except GuardError:
                    # Race: comfy's queue filled between decide and execute.
                    # The VRAM was NOT reclaimed — log and skip the reload.
                    eviction_raced = True
                    self._log("free-raced", {})
                except EngineError as exc:
                    # Comfy unreachable / refused. Same conservative reading
                    # as the race above — the VRAM is NOT confirmed reclaimed,
                    # so the pending load must not be re-triggered into a GPU
                    # that may still be full — and the same per-action
                    # isolation as the unload arm [max-review #9].
                    eviction_raced = True
                    self._log("free-failed", {"error": str(exc)})
                else:
                    # Re-arm the idle TTL so the idle-release rule fires once
                    # per TTL while comfy stays idle, not on every tick.
                    # note_comfy_freed() -> note_freed(resource) (E1 Task 3);
                    # this action type is still comfy-specific pending Task
                    # 6's per-resource action vocabulary, so the resource
                    # name is the same "comfyui" literal `_execute`'s
                    # surrounding block already commits to elsewhere.
                    self._world.note_freed("comfyui")
                    self._log(kind, {})
            elif kind_raw == "noop":
                if action["reason"] == "wont-fit":
                    wont_fit = True
                self._log("noop", {"reason": action["reason"]})

        # After healing a pending load's contention (or finding it already
        # fit), re-trigger the default-route load with its FULL name. Skip if
        # the contention can't be healed (wont-fit) or an eviction raced.
        if pending is not None and not wont_fit and not eviction_raced:
            # Deck-authored load: record BEFORE actuating (same rule as the
            # unload arm above). A failed load then derives 'down' and the
            # reconciler retries under the existing FAILURE_BUDGET —
            # deliberate, not a gap. actor="deck" for the same reason as the
            # unload arm above — this is automatic contention-healing, not
            # an operator's request.
            if self._intent_store is not None:
                self._intent_store.record(
                    LOCAL_LEMONADE_KEY, state="loaded", model=pending["model"], engine="lemonade",
                    actor="deck")
            actuated.add(LOCAL_LEMONADE_KEY)
            try:
                self._lemonade.load(pending["model"])
            except EngineError as exc:
                # Load failed (engine unreachable, bad response, etc.) — log
                # and let the loop survive; the next tick re-evaluates.
                self._log("load-failed", {"error": _error_text(exc)})
            else:
                # Re-arm the failure-dedup memo: a recovery between two
                # IDENTICAL failures must not swallow the second one —
                # fail→recover→fail is the flap the T9-fix review named as
                # this codebase's forbidden dedup class.
                self._last_failure_key = None
                # Deck-initiated load: the model is wanted resident again, so
                # clear any suppression left by a prior deliberate unload.
                self._heal_suppressor.clear()
                # ...and it's a genuine use of the model: record it so the
                # storage watcher's LRU order doesn't treat an auto-reloaded
                # default-route model as "never used".
                if self._catalog is not None:
                    self._catalog.note_used_gguf(
                        pending["model"].removeprefix(_EXTRA_PREFIX))
                self._log("load-retriggered", {"model": pending["model"]})

        return actuated

    # --- lifecycle reconciliation ------------------------------------------

    def _reconcile_pass(self, world: dict, actuated_keys: set[str] = frozenset()) -> None:
        """Compare intent against this tick's observation and restore what
        died. Runs AFTER arbitration on the same snapshot: arbitration
        settles VRAM contention happening right now, reconciliation settles
        desired state over time, and doing it the other way round can
        restore a model arbitration is about to evict (a load/evict flap).

        No-op when no intent store is wired (unit tests and any caller that
        predates the lifecycle work).
        """
        if self._intent_store is None:
            return

        intents = self._intent_store.get()
        observed = merge_observations(
            observe_local(world),
            *self._node_observations(),
        )

        # A key `_execute` actuated THIS tick predates its own action by
        # construction: `world` was snapshotted before `_execute` ran, so
        # observing it now can only see the pre-action state — a successful
        # load-retrigger still reads as unloaded, deriving 'down' and
        # restoring (i.e. loading) it a second time in the same tick, a
        # reconciler fight with the deck's own actuation. Drop it here; the
        # NEXT tick's fresh snapshot reconciles it cleanly.
        for key in actuated_keys:
            observed.pop(key, None)

        statuses = {
            key: derive_status(intents.get(key), obs) for key, obs in observed.items()
        }

        for key, status in statuses.items():
            if status["status"] == "serving":
                self._intent_store.note_healthy(key)
                self._restore_unverified.discard(key)

        actions = plan_reconcile(
            statuses,
            intents,
            auto_enabled=self._policy_store.auto_enabled(),
        )

        for action in actions:
            key = action["key"]

            # A deck-authored load (router route or watcher heal re-trigger)
            # may already be in flight for this exact key: intent was
            # recorded and the engine call started before this tick's world
            # snapshot was taken, so the snapshot still reads unloaded and
            # this derives 'down' even though the deck is already acting on
            # it. Restoring now would double-load the same model from the
            # watcher thread — blocking it up to 180 s and, if that in-flight
            # load's own timeout fires afterward, charging note_failure
            # against what is really a healthy load. Lazy litellm loads never
            # set load_in_flight(), so this can only ever skip deck-authored
            # activity, never mask a real gap. The key reconciles cleanly
            # next tick once the snapshot catches up with the load. A skipped
            # restore is a non-action, so — like the cooldown skip below — it
            # must not be stamped.
            if action["engine"] == "lemonade" and self._lemonade.load_in_flight():
                continue

            last = self._restore_last_attempt_at.get(key)
            now_mono = self._clock()
            if last is not None and now_mono - last < _RESTORE_COOLDOWN_S:
                continue  # a skip is a non-action: no event, no failure charged
            self._restore_last_attempt_at[key] = now_mono
            # Being about to dispatch a restore for a still-unverified key
            # means the previous restore never became healthy: charge it now
            # (after the skip-guards above — skips stay non-actions and never
            # charge). Reuses the lifecycle-restore-failed kind so existing
            # severity mapping applies; no new event vocabulary.
            if key in self._restore_unverified:
                self._restore_unverified.discard(key)
                if self._charge_restore_failure(
                        key, "previous restore never became healthy"):
                    continue  # derive_status reports quarantined next tick
            self._execute_restore(action)

    def _node_observations(self) -> list[dict]:
        """Every swap node's slot observation, in app.observe's vocabulary.

        Translation, TTL cache and backoff live in each node's SparkObserver
        (app.node_clients.NodeObservers), shared with the HTTP paths. This
        adds only the audit trail: the observer parks a probe failure and
        the watcher — which owns the events log — reports it here, once,
        tagged with the node that failed (design §9: existing event kinds
        carry whichever node id acted)."""
        if self._node_observers is None:
            return []
        out = []
        for node_id, observer in sorted(self._node_observers.snapshot().items()):
            status = observer.status()
            error = observer.take_error()
            if error is not None:
                self._log("lifecycle-spark-unreachable",
                          {"node": node_id, "error": error})
            # The agent's serving payload may carry a node-config warning
            # (node-agent serving.py PROBE_URL_WARNING — vllm profiles
            # configured, probe URL unset, detection blind). Agent logs are
            # where signals go to die (invisible for 4 days, 08-12), so
            # surface it in the Events tab — once per (node, warning-text)
            # incident via _node_misconfig_logged. Pre-N1 agents send no
            # such field; absent means nothing to surface. The memo re-arms
            # only on a serving payload OBSERVED without the warning — an
            # unreachable node's cached status carries serving: None, which
            # is "we failed to look", not "we looked and it was gone" (the
            # SparkObserver distinction), so a reachability blip on a node
            # with a standing misconfiguration does not re-log it.
            serving = (status or {}).get("serving")
            warning = (serving or {}).get("warning")
            if warning is not None:
                if self._node_misconfig_logged.get(node_id) != warning:
                    self._node_misconfig_logged[node_id] = warning
                    self._log("lifecycle-node-misconfigured",
                              {"node": node_id, "warning": warning})
            elif serving is not None:
                self._node_misconfig_logged.pop(node_id, None)
            out.append(observe_spark(status, node_id))
        return out

    def _charge_restore_failure(self, key: str, error: str) -> bool:
        """One failure against `key`'s budget: charge, log, and report
        whether this charge quarantined it. Shared by the two sites that
        must agree on the sequence — the unverified-redispatch charge in
        _reconcile_pass and _execute_restore's raise path. `count >=
        FAILURE_BUDGET` is equivalent to the stored `quarantined` flag
        (note_failure sets it exactly there, every reset path clears both
        together, and a missing record returns count 0), so no store
        re-read is needed."""
        count = self._intent_store.note_failure(key)
        self._log("lifecycle-restore-failed",
                  {"key": key, "error": error, "failures": count})
        quarantined = count >= FAILURE_BUDGET
        if quarantined:
            self._log("lifecycle-quarantined", {"key": key})
        return quarantined

    def _execute_restore(self, action: dict) -> None:
        """Perform one restore, recording success or failure against the
        failure budget. A restore that raises must never escape into the
        tick's broad catch — that would skip the remaining actions and hide
        which key failed."""
        key = action["key"]
        try:
            self._restore(action)
        except Exception as exc:  # noqa: BLE001 — per-action isolation, see docstring
            self._charge_restore_failure(key, str(exc))
            return
        # A resource that just came back up has fresh live facts worth
        # capturing, not up to derive_interval_s later — clear the throttle
        # so this tick's already-scheduled _derive_pass() (called right
        # after _reconcile_pass in tick()) runs instead of being skipped. No
        # new I/O here: this only resets an in-memory gate. Floor-limited
        # (_DERIVE_RESTORE_FLOOR_S, see its comment) so a crash-looping
        # resource — _restore() succeeding at the API level without ever
        # raising, charged only cooldown-paced via _restore_unverified
        # until quarantine lands — doesn't turn into a derive on every
        # restore for the whole incident; only the first restore of a
        # burst clears it.
        now_mono = self._clock()
        if (self._last_restore_derive_at is None
                or now_mono - self._last_restore_derive_at >= _DERIVE_RESTORE_FLOOR_S):
            self._last_derive_at = None
            self._last_restore_derive_at = now_mono
        # Dispatched fine, but not yet verified: only a `serving` tick clears
        # this (see _reconcile_pass). If the reconciler comes back to restore
        # this key again first, that redispatch charges the failure budget.
        self._restore_unverified.add(key)
        self._log("lifecycle-restore", {"key": key, "model": action["model"]})

    def _restore(self, action: dict) -> None:
        """Dispatch a restore to the engine that owns the resource.

        Engine dispatch is a mapping of behaviours, not an if-chain over
        resource names, so a new engine is a new entry. The method names are
        the REAL client APIs: hipfire resumes its container
        (app/engines/hipfire.py:162), lemonade loads by name
        (app/engines/lemonade.py:64), spark swaps a profile
        (app/engines/spark.py).

        ComfyUI is deliberately absent, and cannot reach here: app.observe
        maps its 'unknown' state to unreachable, so a down ComfyUI derives
        'unreachable', never 'down'. A handler for it would be dead code
        implying a capability ComfyClient does not have (queue_len/free only).
        """
        engine = action["engine"]

        if engine == "hipfire":
            self._hipfire.resume()
            return
        if engine == "lemonade":
            self._lemonade.load(action["model"])
            return
        if engine == "spark":
            node_id = action["key"].split("/", 1)[0]
            client = (self._node_clients.client_for(node_id)
                      if self._node_clients is not None else None)
            if client is None:
                raise EngineError(
                    f"restore for {action['key']!r} requested but node "
                    f"{node_id!r} is not operable")
            # Spark intent stores the PROFILE in `model` — that is what swap
            # takes, and what observe_spark compares against.
            client.swap(action["model"])
            return

        raise EngineError(f"no restore handler for engine {engine!r}")

    # --- characteristics derivation ----------------------------------------

    def _derive_pass(self) -> None:
        """Refresh derived facts, at most every settings.derive_interval_s.

        Throttled because the tick loop runs every couple of seconds and
        checkpoint directories do not change that fast. Deliberately
        best-effort: one unreadable checkpoint or one unreachable engine
        degrades to fewer facts, never to a failed tick (a raise here is
        still caught by tick()'s own broad supervisor catch, same as every
        other pass).
        """
        now_mono = self._clock()
        if (self._last_derive_at is not None
                and now_mono - self._last_derive_at < self._derive_interval_s):
            return
        self._last_derive_at = now_mono

        if self._on_derive is not None:
            # Test-only seam (see __init__): replaces the real scan below so
            # the throttle timing can be exercised without a real gguf_dir or
            # spark client.
            self._on_derive()
            return

        if self._characteristics_store is None:
            return

        now = datetime.now(UTC).isoformat()

        # is_dir() guards a missing/unmounted /gguf-store: Path.iterdir() on a
        # nonexistent directory raises FileNotFoundError, which would escape
        # this pass into tick()'s supervisor catch every settings.watch_interval
        # seconds instead of degrading to "no local facts" once, here.
        if self._gguf_dir is not None and Path(self._gguf_dir).is_dir():
            for child in sorted(Path(self._gguf_dir).iterdir()):
                if not child.is_dir():
                    continue
                fields = derive_checkpoint(child, now)
                if fields:
                    self._characteristics_store.put_fields(f"model/{child.name}", fields)

        if self._node_observers is not None and self._node_clients is not None:
            for node_id, observer in sorted(self._node_observers.snapshot().items()):
                # Gated on the observer's cached verdict: a down node must
                # not re-add the 5 s timeouts the backoff exists to avoid.
                view = observer.status()
                if view is not None and not view.get("reachable", False):
                    continue
                client = self._node_clients.client_for(node_id)
                if client is None:
                    continue
                try:
                    body = client.models()
                except (EngineError, BusyError):
                    continue   # unreachable: retain last-known facts
                for model_id, fields in derive_live_models(body, now).items():
                    self._characteristics_store.put_fields(
                        f"model/{model_id}", fields)

        self._harvest_catalogs(now)

    def _harvest_catalogs(self, now: str) -> None:
        """Harvest each configured engine's option catalog, once per
        version -- the per-tick loop over ``harvest_catalog_pair``
        (module-level, above), always with ``force=False``: the version
        peek, both `current` gates, and the dedup'd event kinds documented
        on that function all apply unchanged here. This pass must never
        raise past this method: tick()'s broad supervisor catch is not a
        substitute for handling foreseeable failures here (see this
        module's own design notes on that at the top of the file) --
        ``harvest_catalog_pair`` itself upholds that by construction (see
        its own docstring).
        """
        if self._engine_exec is None:
            return

        for node, engine in self._configurable_engines():
            # Same guard as the live-facts loop and _provenance_nodes: a
            # pair whose node has an observer routes through the identical
            # node-agent reads those passes already gate (e.g. the spark
            # pair's harvest_catalog_pair routes to SparkCatalogExec.__call__
            # -> SparkClient.get_catalog(), app/engines/spark.py:399-400).
            # Pairs for nodes with no observer (a future local vLLM) are
            # still harvested every pass.
            observer = (self._node_observers.observer_for(node)
                        if self._node_observers is not None else None)
            if observer is not None:
                view = observer.status()
                if view is not None and not view.get("reachable", False):
                    continue
            harvest_catalog_pair(
                self._engine_exec, self._characteristics_store, self._log,
                node, engine, now, force=False)

    def _configurable_engines(self) -> list[tuple[str, str]]:
        """(node, engine) pairs the Deck can harvest an option catalog from.

        C2: this is now a plain pass-through -- Watcher does no pairing of
        its own. Whatever `configurable_engines` the constructor was handed
        (`self._configurable_engine_pairs`) is returned VERBATIM, pair and
        all; production's real pairs (one per control:"swap" registry node,
        N1) are built once, in app.main's wiring, never here. None becomes
        [] (harvest still opt-in, same shape as
        characteristics_store/hostagent/intent_store).

        C1 truth, live-verified 2026-08-07 -- still true, and still why no
        LOCAL engine appears in production's routes: ``docker exec
        ods-hipfire python3 -c "import vllm"`` -> ``ModuleNotFoundError``.
        hipfire is a Bun/TypeScript daemon (``bun run .hipfire/cli/
        index.ts serve`` + a native daemon binary) -- no Python inference
        stack, nothing argparse to introspect. The argparse-introspection
        probe (app.harvest.PROBE_SOURCE) imports vLLM's own arg parser, so
        it is meaningful for a vLLM-backed engine only: lemonade is
        llama-server, comfyui is comfyui, hipfire is confirmed Bun. C2's
        real targets are each declared swap node's remote vLLM, harvested
        through the node-agent's cached probe (SparkCatalogExec,
        app.engines.spark) rather than a local docker exec -- app.main
        wires them in, this method still knows nothing about it.

        Node vocabulary -- the reason this method takes pairs verbatim
        instead of building them itself: the node half of a pair MUST be a
        node id (``app.observe._LOCAL_NODE``, "local", app/observe.py:28,
        or a swap node's REGISTRY id, e.g. "boxa") -- the same id
        ``GET /api/settings/catalog/{node}/{engine}`` reads into the
        ``engine/{node}/{engine}`` characteristics key (app/routers/
        settings.py:78-82) and ``_resolve`` looks up by (app/routers/
        settings.py:175-183). It must NEVER be ``settings.node_label`` (a
        display string, e.g. "autarch" via MODEL_DECK_NODE_LABEL, shown by
        GET /state -- app/routers/status.py:24) or ``settings.
        spark_node_name`` (a credential-lookup name for the legacy env-seed
        path, app/node_store.py's ``LEGACY_SPARK_SEED_ID`` migration, that
        happened to match the node id on every pre-N1 box) -- that was this
        method's original bug: a harvest that keyed "engine/autarch/hipfire"
        while every reader looks under "engine/local/hipfire", silently
        writing a catalog no API path could ever read. Unit tests never
        caught it because node_label defaults to "local", making label ==
        id. app.main's wiring builds each swap node's pair as ``(that
        node's registry id, "vllm")`` for exactly this reason -- never
        node_label, never spark_node_name.

        `_configurable_engine_pairs` (constructor param `configurable_
        engines`, test-only outside app.main) supplies pairs directly for
        the harvest-scoped unit tests to keep exercising
        _harvest_catalogs's machinery; returned verbatim here (no internal
        pairing step) means the seam itself has no vocabulary decision left
        to get wrong -- a test can hand it `(_LOCAL_NODE, "hipfire")`,
        `("boxa", "vllm")`, or any other pair, and this method cannot
        silently substitute a label for either half.
        """
        return list(self._configurable_engine_pairs or [])

    # --- provenance --------------------------------------------------------

    def _provenance_pass(self) -> None:
        """Refresh the provenance ledger, at most every
        settings.provenance_interval_s.

        RECORDS ONLY. Nothing here pulls, rebuilds, swaps or converges —
        app.reconcile stays the single actuator, and convergence is not
        merely deferred but currently outside the deck's permissions: the
        socket proxy allows start/stop/exec, not pull/create (see
        app/engines/docker_ctl.py's deploy note on why a wildcard there is
        refused), and a swap node's compose files are served read-only by a
        node-agent with no docker access at all.

        Best-effort per source, like _derive_pass: one unreachable container
        marks THAT artifact unavailable — which RETAINS its last known
        version, app.catalog's rule that an unavailable source must not make
        its contents vanish — and never blanks the others. Nothing raises
        past this method; tick()'s broad supervisor catch is not a
        substitute for handling foreseeable failures here.
        """
        if self._provenance_store is None:
            return
        now_mono = self._clock()
        if (self._last_provenance_at is not None
                and now_mono - self._last_provenance_at < self._provenance_interval_s):
            return
        self._last_provenance_at = now_mono

        now = datetime.now(UTC).isoformat()
        # The node ID, never settings.node_label — the display string
        # ("autarch" here) that app/routers/status.py:27 deliberately keeps
        # SEPARATE from the id. Keying artifact ids on a label an operator
        # can rename would orphan every declared origin and its history on
        # a rename, and it left the ledger holding two vocabularies at once
        # (local by label, sparky by id). Same defect the harvest path
        # shipped on 2026-08-07; see _configurable_engines' docstring above.
        node = _LOCAL_NODE

        try:
            self._provenance_local_oci(node, now)
            self._provenance_local_weights(node, now)
            self._provenance_nodes(now)
        except Exception as exc:  # noqa: BLE001 — see docstring
            self._log("provenance-pass-error", {"error": str(exc)})

    def _provenance_local_oci(self, node: str, now: str) -> None:
        """Local engine images, one container inspect each.

        Iterates the park allowlist BY NAME rather than listing containers:
        the socket-proxy allowlist has no `GET /containers/json` rule, and
        this way needs none — it is the same `GET /containers/{name}/json`
        that running()/image_ref() already make.
        """
        if self._dockerctl is None:
            return
        bodies: dict[str, dict | None] = {}
        for name in self._settings.park_allowlist:
            try:
                bodies[name] = self._dockerctl.inspect(name)
            except (EngineError, GuardError, KeyError):
                bodies[name] = None
        # One store round trip for the sweep, not one per artifact
        # [max-review c15]; the returned document also feeds the watch
        # seeding and the retention sweep below without further loads.
        readings = provenance_collect.local_oci_entries(bodies, node)
        observed = {reading["artifact_id"] for reading in readings}
        document = self._provenance_store.observe_all(readings, now=now)
        for reading in readings:
            self._provenance_seed_watch(reading["artifact_id"], now, document)

        # Retention by ABSENCE, not by container name. Identity is the image
        # repository, so a failed inspect no longer tells us which artifact
        # went dark — the body we would have read is the only thing that
        # names its repository. Anything this node previously recorded and
        # did NOT observe this pass is unreachable, which RETAINS its last
        # known version (app.catalog's rule: an unavailable source must not
        # erase what it last told us).
        #
        # Deriving the id from the container name instead would silently
        # no-op the moment a name and its repository differ — and they
        # differ for two of the three allowlist containers today.
        for artifact_id, stored in document.items():
            if artifact_id in observed:
                continue
            if stored.get("kind") == "oci" and stored.get("node") == node:
                self._provenance_store.mark_unavailable(artifact_id, now=now)

    def _provenance_local_weights(self, node: str, now: str) -> None:
        if self._catalog is None:
            return
        self._provenance_store.observe_all(
            provenance_collect.local_file_entries(self._catalog.units(), node),
            now=now)

    def _provenance_nodes(self, now: str) -> None:
        """Every swap node's engine images, from the two node-agent reads
        that already exist (compose text for the reference, the harvested
        catalog for the digest). Keys ride the node's REGISTRY id — the
        same id every other vocabulary uses."""
        if self._node_observers is None or self._node_clients is None:
            return
        for node_id, observer in sorted(self._node_observers.snapshot().items()):
            view = observer.status()
            if view is not None and not view.get("reachable", False):
                continue
            client = self._node_clients.client_for(node_id)
            if client is None:
                continue
            try:
                profiles = [p["name"] for p in client.status()["profiles"]]
            except (EngineError, BusyError, GuardError, KeyError, TypeError):
                continue
            texts: dict[str, str] = {}
            for profile in profiles:
                try:
                    texts[profile] = client.get_compose(profile)
                except (EngineError, BusyError, GuardError):
                    continue
            try:
                catalog = client.get_catalog().get("catalog")
            except (EngineError, BusyError, GuardError, AttributeError):
                catalog = None
            readings = provenance_collect.spark_oci_entries(texts, catalog, node_id)
            document = self._provenance_store.observe_all(readings, now=now)
            for reading in readings:
                self._provenance_seed_watch(reading["artifact_id"], now, document)

    def _provenance_seed_watch(self, artifact_id: str, now: str,
                               document: dict[str, dict]) -> None:
        """A derivable origin gains its watch entry without an operator
        typing it (Task 9's ``provenance_collect.merge_seeded_watch``).

        Called for every oci artifact this pass just observed above — NOT
        for weights (``_provenance_local_weights``): ``seed_watch`` only
        ever derives from an oci origin, so calling this there would only
        ever recompute ``[]`` against ``[]``, one extra lookup per catalog
        unit for no possible effect.

        ``merge_seeded_watch`` needs the STORED entry, not the one this
        pass just built: the collector's own entry dict (``local_oci_entries``
        / ``spark_oci_entries``) carries no ``origin`` at all (that is
        exclusively operator-declared, via ``PUT /origin``) and no ``watch``.
        ``document`` — the document ``observe_all`` just SAVED, handed back
        precisely so this seeding needs no per-artifact re-load [max-review
        c15] — has both. A concurrent ``PUT /origin`` landing between that
        save and this call is the same race the old fresh ``store.entry()``
        read had, only narrower, and ``set_watch``'s own locked no-op check
        remains the authority either way.

        DOUBLE PROTECTION AGAINST HISTORY SPAM ON UNCHANGED DATA, and they
        are not equally load-bearing. ``ProvenanceStore.set_watch`` is the
        authoritative one: an unchanged watch list (compared order-
        independently by id) is a true no-op there — no write, no history —
        by design, precisely because this task calls it every pass with the
        same freshly-computed sources (see its docstring). The equality
        check below is defense-in-depth only: it saves the lock/load/save
        round trip on the overwhelmingly common "nothing changed" case, but
        if it ever missed a spurious difference (e.g. dict key order — none
        exists today since ``merge_seeded_watch`` derives at most one
        source), correctness would still hold because ``set_watch`` itself
        would no-op.
        """
        entry = document.get(artifact_id)
        if entry is None:
            return
        merged = provenance_collect.merge_seeded_watch(entry)
        if merged == entry["watch"]:
            return
        try:
            self._provenance_store.set_watch(artifact_id, merged, now)
        except ValueError as exc:
            # BEST-EFFORT PER ARTIFACT, as this pass's docstring promises.
            # `set_watch` validates every source (app/provenance.py:452), and
            # a hand-edited `provenance.json` can hold one no write path could
            # have produced -- BadWatch and BadArtifactId are both
            # ValueErrors. Unhandled, that escaped into _provenance_pass's
            # catch-all above and abandoned everything after this artifact:
            # local weights AND the rest of the _provenance_nodes sweep,
            # once per provenance_interval_s, forever. Logged, never
            # swallowed -- nothing else on this path can tell the operator
            # their file is unusable.
            self._log("provenance-seed-watch-failed",
                      {"artifact_id": artifact_id, "error": str(exc)})

    # Event kinds whose consecutive identical repeats are collapsed to a
    # single log line, so a persistent state (a stuck 'wont-fit', a crashing
    # tick, a comfy queue that keeps racing) doesn't spam the audit log every
    # tick. A different kind — or the same kind with a different detail — in
    # between resets the suppression. Real one-shot actions (unloads, frees,
    # loads) are NEVER deduped and always logged.
    _DEDUP_KINDS = frozenset({"noop", "tick-error", "free-raced", "host-agent-busy",
                          "lifecycle-spark-unreachable", "harvest-empty",
                          "harvest-failed", "provenance-seed-watch-failed"})

    # load/unload failures get their OWN memo, not the one-slot
    # _last_event_key: a contention tick interleaves free_comfyui with every
    # load-failed, so the global "same as the immediately-previous event"
    # check never fires for exactly the outage that spams — one degraded
    # backend writing an identical failure line every 2 s tick, the
    # reachable trim-thrash input [T10 review]. The success arms clear this
    # memo so a fail→recover→identical-fail flap logs both failures (the
    # dedup-key-never-cleared class the T9-fix review forbade).
    _FAILURE_DEDUP_KINDS = frozenset({"load-failed", "unload-failed"})

    def _log(self, kind: str, detail: dict) -> None:
        key = (kind, tuple(sorted(detail.items())))
        if kind in self._FAILURE_DEDUP_KINDS:
            if key == self._last_failure_key:
                return
            self._last_failure_key = key
        elif kind in self._DEDUP_KINDS and key == self._last_event_key:
            return
        log_event(self._events_path, kind, detail)
        self._last_event_key = key
