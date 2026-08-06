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

from app.derive_checkpoint import derive_checkpoint
from app.derive_live import derive_live_models
from app.engines import BusyError, EngineError, GuardError
from app.events import log_event
from app.lifecycle import derive_status
from app.observe import (
    SparkObserver,
    merge_observations,
    observe_local,
    observe_spark,
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
        # Test-only seam: when set, called once per non-throttled derive pass
        # INSTEAD of the real checkpoint/engine scan, so the throttle timing
        # itself can be tested without a real gguf_dir or spark client.
        # Always None in production (app.main never passes it).
        self._on_derive = on_derive

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
            self._execute(actions, pending)
            self._reconcile_pass(world)
            self._derive_pass()
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

    def _execute(self, actions: list[dict], pending: dict | None) -> None:
        wont_fit = False
        eviction_raced = False

        for action in actions:
            kind = action["type"]
            if kind == "unload_lemonade":
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

    # --- lifecycle reconciliation ------------------------------------------

    def _reconcile_pass(self, world: dict) -> None:
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

        for client in self._live_fact_sources().values():
            try:
                body = client.models()
            except (EngineError, BusyError):
                continue   # unreachable: retain last-known facts
            for model_id, fields in derive_live_models(body, now).items():
                self._characteristics_store.put_fields(f"model/{model_id}", fields)

    def _live_fact_sources(self) -> dict:
        """Engine clients exposing an OpenAI-style /v1/models surface.
        Spark is the one that matters — it is the only source of truth for
        remote model facts until a node-agent increment lands."""
        sources = {}
        if self._spark is not None:
            sources["spark"] = self._spark
        return sources

    # Event kinds whose consecutive identical repeats are collapsed to a
    # single log line, so a persistent state (a stuck 'wont-fit', a crashing
    # tick, a comfy queue that keeps racing) doesn't spam the audit log every
    # tick. A different kind — or the same kind with a different detail — in
    # between resets the suppression. Real one-shot actions (unloads, frees,
    # loads) are NEVER deduped and always logged.
    _DEDUP_KINDS = frozenset({"noop", "tick-error", "free-raced", "host-agent-busy",
                          "lifecycle-spark-unreachable"})

    def _log(self, kind: str, detail: dict) -> None:
        key = (kind, tuple(sorted(detail.items())))
        if kind in self._DEDUP_KINDS and key == self._last_event_key:
            return
        log_event(self._events_path, kind, detail)
        self._last_event_key = key
