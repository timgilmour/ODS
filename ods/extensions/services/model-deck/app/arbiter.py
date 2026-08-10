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

Action dicts (the only three shapes ``decide`` ever returns):
    {"type": "unload_lemonade", "model": <str>}
    {"type": "free_comfyui"}
    {"type": "noop", "reason": <str>}   # reason in {"fits", "wont-fit"}

Design notes that are load-bearing (and tested):

* IDLE RELEASE (rule 1) is allowed on the default-route lemonade model. That
  is the deliberate idle-GPU-burn fix — a resident-but-idle llama.cpp model
  pins a card at 100%/86 W doing nothing; unloading drops it to ~17 W and the
  next request reloads it in ~4 s. This exception applies ONLY to idle
  release, never to contention.

* CONTENTION HEALING (rule 2) NEVER evicts the default-route lemonade model,
  no matter how much VRAM that would free. It also never touches hipfire
  (which lives on a different GPU and is pinned) and never frees a busy or
  unknown-state comfy queue. Candidates on the pending GPU are sorted
  ascending by ``policy[tenant]["priority"]`` (lowest priority evicted
  first) — with the default policies (comfyui 40, lemonade 50) this
  reproduces the historical comfyui-then-lemonade order. Because on this box
  lemonade and comfyui share the one GPU that pending loads target,
  ``decide`` doesn't need per-GPU attribution — the pending GPU's candidates
  are always {comfyui, lemonade}, and hipfire is structurally excluded.

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

* The watcher YIELDS to an in-flight set apply: if ``app.sets`` holds its
  apply lock, ``tick()`` returns immediately (no snapshot, no actions) so the
  two never interleave real evictions/loads on the live box.

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
  resource can have ``_restore()`` succeed at the API level every ~2 s tick
  without ever raising, so the failure-budget/quarantine machinery (which
  only trips on a raise) never bounds it — without the floor every one of
  those ticks would re-clear the throttle and re-run the scan.
  ``characteristics_store=None`` (the default) disables the whole pass —
  same opt-in shape as ``hostagent``/``intent_store``.
"""

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from app import provenance_collect
from app.derive_checkpoint import derive_checkpoint
from app.derive_live import derive_live_models
from app.engines import BusyError, EngineError, GuardError
from app.events import log_event
from app.harvest import PROBE_INTERPRETER, PROBE_SOURCE, parse_probe_output
from app.lifecycle import derive_status
from app.observe import (
    _LOCAL_NODE,
    LOCAL_LEMONADE_KEY,
    SparkObserver,
    merge_observations,
    observe_local,
    observe_spark,
    spark_node_id,
)
from app.reconcile import plan_reconcile
from app.sets import apply_in_progress

# VRAM overhead slack subtracted when estimating comfyui's reclaimable bytes
# from raw GPU usage (fragmentation, driver/runtime overhead, small tenants).
_SLACK_BYTES = 1024**3  # 1 GiB

_EXTRA_PREFIX = "extra."

# Minimum spacing between restore-triggered characteristics-derive-throttle
# clears (see Watcher._execute_restore). A crash-looping resource can have
# _restore() succeed at the API level every ~2 s tick (e.g. resume() returns
# fine) while the process dies again before the next tick — the failure-
# budget/quarantine machinery only trips when _restore() RAISES, so it does
# not bound this case. Without a floor, every one of those ticks would clear
# the derive throttle and re-run the checkpoint scan + spark probe, exactly
# the pointless per-tick I/O the throttle exists to prevent, concentrated
# during an active incident. 30 s bounds that to at most one extra derive per
# half-minute of crash-looping — well above the ~2 s tick so one flap cycle
# can't retrigger it, and well below settings.derive_interval_s's default
# 300 s so a genuinely new incident more than 30 s later still gets its own
# immediate derive.
_DERIVE_RESTORE_FLOOR_S = 30.0

# Minimum spacing between restore attempts FOR THE SAME KEY (storm limiter,
# defense-in-depth for observation windows the earlier tasks don't cover —
# deck-invisible lazy loads, hipfire/spark record-last windows). The failure
# budget (app.intent.FAILURE_BUDGET) quarantines repeat FAILURES, but a
# restore that "succeeds" at the API level and dies out-of-band before the
# next tick never raises, so it never charges the budget — without this
# floor, reconcile re-dispatches a real restore every ~2 s tick for as long
# as the resource keeps flapping (observed live: three restores in 10 s,
# 2026-08-06). A skipped restore is a non-action: no event, no failure
# charged.
_RESTORE_COOLDOWN_S = 30.0


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
    actions: list[dict] = []
    tenants = world["tenants"]

    lem = tenants["lemonade"]
    lem_pol = policy["lemonade"]
    if (
        lem["state"] == "loaded"
        and not lem_pol["pinned"]
        and lem_pol["idle_ttl"] > 0
        and lem["idle_s"] is not None
        and lem["idle_s"] >= lem_pol["idle_ttl"]
    ):
        # NOTE: the default-route model is intentionally NOT guarded here —
        # idle release on it is the idle-GPU-burn fix (reload ~4 s).
        actions.append({"type": "unload_lemonade", "model": lem["model"]})

    comfy = tenants["comfyui"]
    comfy_pol = policy["comfyui"]
    if (
        comfy["state"] == "idle"
        and comfy["queue"] == 0
        and not comfy_pol["pinned"]
        and comfy_pol["idle_ttl"] > 0
        and comfy["idle_s"] is not None
        and comfy["idle_s"] >= comfy_pol["idle_ttl"]
        # A free that can't reclaim anything is a no-op that re-arms the TTL
        # and re-fires every idle_ttl seconds forever, flooding the event
        # ring. None (comfy's GPU unresolvable) must not suppress the free:
        # unknown usage is not proof there's nothing to reclaim.
        and _comfy_reclaimable(world) != 0
    ):
        actions.append({"type": "free_comfyui"})

    return actions


def _comfy_reclaimable(world: dict) -> int | None:
    """Estimated bytes a comfy free would reclaim, or None when comfy's GPU
    isn't in the snapshot. Same estimate as ``_eviction_candidates``: comfy's
    VRAM presence isn't directly observable, so attribute to it whatever its
    GPU's usage doesn't explain — minus a co-resident loaded lemonade's
    footprint and the fixed slack allowance."""
    placement = world.get("placement") or {}
    gpu = _find_gpu(world["gpus"], placement.get("comfyui"))
    if gpu is None:
        return None

    lem = world["tenants"]["lemonade"]
    lem_footprint = (
        lem["footprint"]
        if (
            lem["state"] == "loaded"
            and lem["footprint"]
            and placement.get("lemonade") == placement.get("comfyui")
        )
        else 0
    )
    return max(0, gpu["used"] - lem_footprint - _SLACK_BYTES)


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
    """Evictable tenants on the pending GPU as ``(action, reclaimable_bytes)``,
    sorted ascending by ``policy[tenant]["priority"]`` (lowest priority
    evicted first). Eligibility guards are unchanged from before; only the
    resulting order is policy-driven.

    hipfire is never a candidate (it lives on the other GPU and is pinned) —
    the guard against touching it is by structural omission here.
    """
    tenants = world["tenants"]
    lem = tenants["lemonade"]
    comfy = tenants["comfyui"]

    lem_loaded_footprint = (
        lem["footprint"] if lem["state"] == "loaded" and lem["footprint"] else 0
    )

    candidates: list[tuple[str, dict, int]] = []

    # comfyui — never free a busy/unknown queue, never if pinned. Its VRAM
    # presence isn't in world.tenants, so estimate reclaimable from the GPU
    # usage gap (per plan): used - lemonade-footprint-if-loaded - slack.
    if (
        not policy["comfyui"]["pinned"]
        and comfy["state"] == "idle"
        and comfy["queue"] == 0
    ):
        reclaimable = max(0, gpu["used"] - lem_loaded_footprint - _SLACK_BYTES)
        candidates.append(("comfyui", {"type": "free_comfyui"}, reclaimable))

    # lemonade — never evict the default-route model (ABSOLUTE for rule 2),
    # never if pinned, and only if its footprint is known (else unquantifiable).
    if (
        lem["state"] == "loaded"
        and not policy["lemonade"]["pinned"]
        and lem["model"] != world["default_route"]
        and lem["footprint"]
    ):
        candidates.append(
            ("lemonade", {"type": "unload_lemonade", "model": lem["model"]}, lem["footprint"])
        )

    candidates.sort(key=lambda c: policy[c[0]]["priority"])
    return [(action, reclaimable) for _, action, reclaimable in candidates]


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
        spark=None,
        spark_observer=None,
        characteristics_store=None,
        gguf_dir=None,
        clock=time.monotonic,
        on_derive=None,
        engine_exec=None,
        configurable_engines=None,
        provenance_store=None,
        dockerctl=None,
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
        # Remote single-slot node. None on a box with no spark configured —
        # observe_spark then emits no key at all, so an undeclared resource
        # can never appear as a phantom failure.
        self._spark = spark
        # Shared with the HTTP routers when app.main wires it (one cache for
        # the process); self-built otherwise so a bare Watcher still works.
        self._spark_observer = (
            spark_observer if spark_observer is not None
            else SparkObserver(lambda: self._spark)
        )
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
        # builds its one real pair with `(spark_node_id(), "vllm")`, never
        # `settings.node_label`/`settings.spark_node_name` -- see
        # _configurable_engines' docstring for the historical bug that rule
        # guards against. None (every caller except the harvest-scoped unit
        # tests and app.main) leaves _configurable_engines returning [].
        self._configurable_engine_pairs = configurable_engines
        # Provenance ledger (app.provenance). None -- every unit test and
        # every pre-provenance caller -- disables the pass entirely, the same
        # opt-in shape as characteristics_store/intent_store/hostagent above.
        # `dockerctl` is the LOCAL oci source; the spark client (already held)
        # is the remote one; the catalog (already held) is weights. The pass
        # RECORDS ONLY -- see _provenance_pass.
        self._provenance_store = provenance_store
        self._dockerctl = dockerctl
        self._provenance_interval_s = settings.provenance_interval_s
        self._last_provenance_at: float | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_event_key = None

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
        # Yield to an in-flight set apply: the two must never interleave real
        # evictions/loads on the live box. Checked WITHOUT acquiring the lock
        # (a peek), so a running apply makes this tick a clean no-op — no
        # snapshot, no actions.
        if apply_in_progress():
            return

        # DELIBERATE broad catch: this is a supervisor loop. A crash in any
        # single tick (malformed engine body, transient client bug, a bad
        # snapshot) must NOT take the whole arbiter down — loop survival
        # trumps the house 'let it crash' default HERE, and only here. The
        # error is logged so it's never silent.
        try:
            gpus = self._read_gpus(self._settings.drm_root, self._settings.kfd_root)
            world = self._world.snapshot(
                gpus, self._lemonade, self._comfy, self._hipfire, self._litellm, self._registry
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
            actuated_keys = self._execute(actions, pending)
            self._reconcile_pass(world, actuated_keys)
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
            kind = action["type"]
            if kind == "unload_lemonade":
                # Whoever actuates, records — and records FIRST, so a tick
                # that lands mid-unload derives 'parked', never 'down'
                # (2026-08-06).
                if self._intent_store is not None:
                    self._intent_store.record(
                        LOCAL_LEMONADE_KEY, state="unloaded", model=None, engine="lemonade")
                # Added BEFORE the engine call (mirrors the retrigger tail
                # below) so the invariant is structural: an unload wrapped in
                # try/except in the future still can't return without this
                # key marked actuated.
                actuated.add(LOCAL_LEMONADE_KEY)
                self._lemonade.unload(action["model"])
                # Deck-initiated unload (idle release OR contention eviction):
                # arm suppression so healing can't immediately revert it.
                self._heal_suppressor.note_deck_unload()
                self._log(kind, {"model": action["model"]})
            elif kind == "free_comfyui":
                try:
                    self._comfy.free()
                except GuardError:
                    # Race: comfy's queue filled between decide and execute.
                    # The VRAM was NOT reclaimed — log and skip the reload.
                    eviction_raced = True
                    self._log("free-raced", {})
                else:
                    # Re-arm the idle TTL so the idle-release rule fires once
                    # per TTL while comfy stays idle, not on every tick.
                    self._world.note_comfy_freed()
                    self._log(kind, {})
            elif kind == "noop":
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
            # deliberate, not a gap.
            if self._intent_store is not None:
                self._intent_store.record(
                    LOCAL_LEMONADE_KEY, state="loaded", model=pending["model"], engine="lemonade")
            actuated.add(LOCAL_LEMONADE_KEY)
            try:
                self._lemonade.load(pending["model"])
            except EngineError as exc:
                # Load failed (engine unreachable, bad response, etc.) — log
                # and let the loop survive; the next tick re-evaluates.
                self._log("load-failed", {"error": str(exc)})
            else:
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
        spark_status = self._spark_status()
        observed = merge_observations(
            observe_local(world),
            observe_spark(spark_status),
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
            self._execute_restore(action)

    def _spark_status(self) -> dict | None:
        """The spark node's state in app.observe's vocabulary, or None when
        no spark is configured.

        The translation and its TTL cache live in app.observe.SparkObserver,
        shared with the HTTP paths so a tick and a GET cost one probe. All
        this adds is the audit trail: the observer parks a probe failure and
        the watcher — which owns the events log — reports it here, once,
        wherever the failing probe actually ran.
        """
        status = self._spark_observer.status()
        error = self._spark_observer.take_error()
        if error is not None:
            self._log("lifecycle-spark-unreachable", {"error": error})
        return status

    def _execute_restore(self, action: dict) -> None:
        """Perform one restore, recording success or failure against the
        failure budget. A restore that raises must never escape into the
        tick's broad catch — that would skip the remaining actions and hide
        which key failed."""
        key = action["key"]
        try:
            self._restore(action)
        except Exception as exc:  # noqa: BLE001 — per-action isolation, see docstring
            count = self._intent_store.note_failure(key)
            self._log("lifecycle-restore-failed",
                      {"key": key, "error": str(exc), "failures": count})
            if self._intent_store.get().get(key, {}).get("quarantined"):
                self._log("lifecycle-quarantined", {"key": key})
            return
        # A resource that just came back up has fresh live facts worth
        # capturing, not up to derive_interval_s later — clear the throttle
        # so this tick's already-scheduled _derive_pass() (called right
        # after _reconcile_pass in tick()) runs instead of being skipped. No
        # new I/O here: this only resets an in-memory gate. Floor-limited
        # (_DERIVE_RESTORE_FLOOR_S, see its comment) so a crash-looping
        # resource — _restore() succeeding every ~2 s tick without ever
        # raising, so the failure-budget/quarantine above never trips —
        # doesn't turn into a derive on every tick for the whole incident;
        # only the first restore of a burst clears it.
        now_mono = self._clock()
        if (self._last_restore_derive_at is None
                or now_mono - self._last_restore_derive_at >= _DERIVE_RESTORE_FLOOR_S):
            self._last_derive_at = None
            self._last_restore_derive_at = now_mono
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
            if self._spark is None:
                raise EngineError(
                    "spark restore requested but no spark client is configured")
            # Spark intent stores the PROFILE in `model` — that is what swap
            # takes, and what observe_spark compares against.
            self._spark.swap(action["model"])
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

        for name, client in self._live_fact_sources().items():
            # Spark alone is gated on the observer's cached verdict: it is
            # the one source with a backoff (SparkObserver) worth respecting
            # — a direct client.models() call here would block on the same
            # 5 s timeouts the backoff exists to avoid. Keyed on `name`, not
            # applied to the whole loop, so a future non-spark source (see
            # this method's docstring) still gets probed every pass even
            # while spark is down.
            if name == "spark":
                view = self._spark_observer.status()
                if view is not None and not view.get("reachable", False):
                    continue
            try:
                body = client.models()
            except (EngineError, BusyError):
                continue   # unreachable: retain last-known facts
            for model_id, fields in derive_live_models(body, now).items():
                self._characteristics_store.put_fields(f"model/{model_id}", fields)

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
            harvest_catalog_pair(
                self._engine_exec, self._characteristics_store, self._log,
                node, engine, now, force=False)

    def _configurable_engines(self) -> list[tuple[str, str]]:
        """(node, engine) pairs the Deck can harvest an option catalog from.

        C2: this is now a plain pass-through -- Watcher does no pairing of
        its own. Whatever `configurable_engines` the constructor was handed
        (`self._configurable_engine_pairs`) is returned VERBATIM, pair and
        all; production's one real pair is built once, in app.main's
        wiring, never here. None becomes [] (harvest still opt-in, same
        shape as characteristics_store/hostagent/intent_store).

        C1 truth, live-verified 2026-08-07 -- still true, and still why no
        LOCAL engine appears in production's routes: ``docker exec
        ods-hipfire python3 -c "import vllm"`` -> ``ModuleNotFoundError``.
        hipfire is a Bun/TypeScript daemon (``bun run .hipfire/cli/
        index.ts serve`` + a native daemon binary) -- no Python inference
        stack, nothing argparse to introspect. The argparse-introspection
        probe (app.harvest.PROBE_SOURCE) imports vLLM's own arg parser, so
        it is meaningful for a vLLM-backed engine only: lemonade is
        llama-server, comfyui is comfyui, hipfire is confirmed Bun. C2's
        one real target is the spark node's remote vLLM, harvested through
        the node-agent's cached probe (SparkCatalogExec, app.engines.spark)
        rather than a local docker exec -- app.main wires it in, this
        method still knows nothing about it.

        Node vocabulary -- the reason this method takes pairs verbatim
        instead of building them itself: the node half of a pair MUST be a
        node id (``app.observe._LOCAL_NODE``, "local", app/observe.py:28,
        or ``app.observe.spark_node_id()``, "sparky") -- the same id
        ``GET /api/settings/catalog/{node}/{engine}`` reads into the
        ``engine/{node}/{engine}`` characteristics key (app/routers/
        settings.py:78-82) and ``_resolve`` looks up by (app/routers/
        settings.py:175-183). It must NEVER be ``settings.node_label`` (a
        display string, e.g. "autarch" via MODEL_DECK_NODE_LABEL, shown by
        GET /state -- app/routers/status.py:24) or ``settings.
        spark_node_name`` (a credential-lookup name that happens to match
        the node id today -- see ``spark_node_id``'s own docstring) --
        that was this method's original bug: a harvest that keyed
        "engine/autarch/hipfire" while every reader looks under
        "engine/local/hipfire", silently writing a catalog no API path
        could ever read. Unit tests never caught it because node_label
        defaults to "local", making label == id. app.main's wiring builds
        spark's pair as ``(spark_node_id(), "vllm")`` for exactly this
        reason -- never node_label, never spark_node_name.

        `_configurable_engine_pairs` (constructor param `configurable_
        engines`, test-only outside app.main) supplies pairs directly for
        the harvest-scoped unit tests to keep exercising
        _harvest_catalogs's machinery; returned verbatim here (no internal
        pairing step) means the seam itself has no vocabulary decision left
        to get wrong -- a test can hand it `(_LOCAL_NODE, "hipfire")`,
        `(spark_node_id(), "vllm")`, or any other pair, and this method
        cannot silently substitute a label for either half.
        """
        return list(self._configurable_engine_pairs or [])

    def _live_fact_sources(self) -> dict:
        """Engine clients exposing an OpenAI-style /v1/models surface.
        Spark is the one that matters — it is the only source of truth for
        remote model facts until a node-agent increment lands."""
        sources = {}
        if self._spark is not None:
            sources["spark"] = self._spark
        return sources

    # --- provenance --------------------------------------------------------

    def _provenance_pass(self) -> None:
        """Refresh the provenance ledger, at most every
        settings.provenance_interval_s.

        RECORDS ONLY. Nothing here pulls, rebuilds, swaps or converges —
        app.reconcile stays the single actuator, and convergence is not
        merely deferred but currently outside the deck's permissions: the
        socket proxy allows start/stop/exec, not pull/create (see
        app/engines/docker_ctl.py's deploy note on why a wildcard there is
        refused), and sparky's compose files are served read-only by a
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
            self._provenance_spark(now)
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
        observed: set[str] = set()
        for entry in provenance_collect.local_oci_entries(bodies, node):
            observed.add(entry["artifact_id"])
            current = entry.pop("current")
            self._provenance_store.observe(**entry, current=current, now=now)
            self._provenance_seed_watch(entry["artifact_id"], now)

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
        for artifact_id, stored in self._provenance_store.get().items():
            if artifact_id in observed:
                continue
            if stored.get("kind") == "oci" and stored.get("node") == node:
                self._provenance_store.mark_unavailable(artifact_id, now=now)

    def _provenance_local_weights(self, node: str, now: str) -> None:
        if self._catalog is None:
            return
        for entry in provenance_collect.local_file_entries(self._catalog.units(), node):
            current = entry.pop("current")
            self._provenance_store.observe(**entry, current=current, now=now)

    def _provenance_spark(self, now: str) -> None:
        """Sparky's engine images, from the two node-agent reads that already
        exist (compose text for the reference, the harvested catalog for the
        digest). ``spark_node_id()`` — NEVER settings.spark_node_name, a
        coincidentally-equal different string; see _configurable_engines'
        docstring for the live-only bug that rule guards against."""
        if self._spark is None:
            return
        # The observer's cached verdict, no fresh probe: while sparky is down
        # every direct call below would block on 5 s timeouts the backoff
        # exists to avoid — and the pass would learn nothing anyway.
        view = self._spark_observer.status()
        if view is not None and not view.get("reachable", False):
            return
        node = spark_node_id()
        try:
            profiles = [p["name"] for p in self._spark.status()["profiles"]]
        except (EngineError, BusyError, GuardError, KeyError, TypeError):
            return
        texts: dict[str, str] = {}
        for profile in profiles:
            try:
                texts[profile] = self._spark.get_compose(profile)
            except (EngineError, BusyError, GuardError):
                continue
        try:
            catalog = self._spark.get_catalog().get("catalog")
        except (EngineError, BusyError, GuardError, AttributeError):
            catalog = None
        for entry in provenance_collect.spark_oci_entries(texts, catalog, node):
            current = entry.pop("current")
            self._provenance_store.observe(**entry, current=current, now=now)
            self._provenance_seed_watch(entry["artifact_id"], now)

    def _provenance_seed_watch(self, artifact_id: str, now: str) -> None:
        """A derivable origin gains its watch entry without an operator
        typing it (Task 9's ``provenance_collect.merge_seeded_watch``).

        Called for every oci artifact this pass just observed above — NOT
        for weights (``_provenance_local_weights``): ``seed_watch`` only
        ever derives from an oci origin, so calling this there would only
        ever recompute ``[]`` against ``[]``, one extra store read per
        catalog unit for no possible effect.

        ``merge_seeded_watch`` needs the STORED entry, not the one this
        pass just built: the collector's own entry dict (``local_oci_entries``
        / ``spark_oci_entries``) carries no ``origin`` at all (that is
        exclusively operator-declared, via ``PUT /origin``) and no ``watch``
        by the time ``current`` has been popped off it for ``observe()``.
        Only ``store.entry()`` — read fresh, after ``observe()`` above has
        landed this pass's identity — has both.

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
        entry = self._provenance_store.entry(artifact_id)
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
            # local weights AND the whole sparky sweep, once per
            # provenance_interval_s, forever. Logged, never swallowed --
            # nothing else on this path can tell the operator their file is
            # unusable.
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

    def _log(self, kind: str, detail: dict) -> None:
        key = (kind, tuple(sorted(detail.items())))
        if kind in self._DEDUP_KINDS and key == self._last_event_key:
            return
        log_event(self._events_path, kind, detail)
        self._last_event_key = key
