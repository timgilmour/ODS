"""The update-checking pass.

READS UPSTREAMS, WRITES A RECORD. Nothing here pulls, builds, swaps or
converges -- app.reconcile stays the single actuator, and convergence is not
merely deferred but outside the deck's permissions today (provenance design
D11). A pass that only reads and records cannot become a second actuator, so
that invariant holds structurally rather than by convention.

NEVER CALLED FROM arbiter.Watcher.tick(). That tick is one synchronous thread
running reconcile -> derive -> provenance in sequence; a network call there
would stall the reconciler, which is what keeps models alive. This pass runs
on its own thread (see UpdateChecker) and hands off through ProvenanceStore's
existing lock and atomic writes.

EVENTS ARE LOGGED ON TRANSITION. A pending update that re-logged every pass
would be exactly the spam app/arbiter.py:438 already calls out.
"""

import threading

import httpx

from app import events, updates
from app.updates import git as git_checks
from app.updates import oci as oci_checks
from app.updates.fetch import make_fetch

_DISPATCH = {
    "oci_channel": oci_checks.check_channel,
    "oci_tags": oci_checks.check_tags,
    "git_compare": git_checks.check_compare,
    "git_tags": git_checks.check_tags,
}


def dispatch(source: dict, fetch) -> dict:
    """Route one watch source to its checker. Any failure becomes an
    UNAVAILABLE result for THAT source -- siblings are unaffected, and the
    pass never raises."""
    if not isinstance(source, dict):
        # DEVIATION, ruled Important by the coordinator (task-6-report.md's
        # second "Fix round"): nothing may execute before this guard.
        # `source.get("check")` on the very next line raised AttributeError
        # for anything that isn't a dict, and that escapes this function's
        # own try/except entirely (the exception happens before the `try`
        # even starts), taking down the whole `run_pass` call.
        #
        # WHAT MAKES IT REACHABLE TODAY IS THE DIRECT CALL, not stored data.
        # `dispatch` is public and `dispatch(None, fetch)` is a supported
        # call. The original motivation -- a hand-edited `provenance.json`
        # watch list holding a bare string or number -- no longer reaches
        # here: `provenance._stored_list` drops non-object elements at the
        # read boundary (app/provenance.py:80-85, Task 5 round 4). The guard
        # stays because the contract it defends is `dispatch`'s own, not the
        # store's. Same fallback shape the "no checker" branch below returns.
        return {"id": "?", "status": updates.UNAVAILABLE, "current": None,
                "latest": None, "detail": {},
                "note": f"watch source is not a mapping: {type(source).__name__}"}
    checker = _DISPATCH.get(source.get("check"))
    if checker is None:
        return {"id": source.get("id", "?"), "status": updates.UNAVAILABLE,
                "current": source.get("pinned"), "latest": None, "detail": {},
                "note": f"no checker for {source.get('check')!r}"}
    try:
        return checker(source, fetch)
    except Exception as exc:  # noqa: BLE001 — per-source isolation, see docstring
        # DEVIATION FROM THE BRIEF (see task-6-report.md for the full
        # writeup): two changes from the reference implementation, both
        # confirmed by a failing test before this fix landed.
        #
        # 1. `source.get("id", "?")`, not `source["id"]`. Every checker
        #    builds its result via `source["id"]` (git.py's and oci.py's
        #    `_result` helpers), so a source missing "id" makes the checker
        #    raise KeyError -- which is exactly what this except exists to
        #    catch. The brief's except-handler then ALSO did `source["id"]`
        #    to build the fallback, raising the identical KeyError and
        #    letting it escape dispatch (and run_pass, and every sibling
        #    artifact in the same pass) -- the one thing this function's own
        #    docstring says can never happen. app.provenance.record_update
        #    already defends against a stored source missing "id"; dispatch
        #    must match that discipline, and the "no checker" branch two
        #    lines up already does.
        #
        # 2. `type(exc).__name__` only, never `str(exc)`, in the note used
        #    for the failed-event dedup key (_log_transitions, below). A
        #    raised exception's message can be non-deterministic across
        #    calls for the IDENTICAL underlying failure -- the standard
        #    example is urllib3/requests wrapping a socket error in a
        #    connection-object repr that embeds a memory address, which
        #    differs every call. Keying the dedup on that text would defeat
        #    it entirely: a permanently-broken remote would log a new
        #    "update-check-failed" event every single pass, which is
        #    precisely the spam app/arbiter.py:438 was fixed to avoid by
        #    logging the exception's CLASS only -- a precedent this module's
        #    own docstring cites by line number.
        return {"id": source.get("id", "?"), "status": updates.UNAVAILABLE,
                "current": source.get("pinned"), "latest": None, "detail": {},
                "note": f"checker raised {type(exc).__name__}"}


def run_pass(store, fetch, events_path, *, dedup: dict) -> dict:
    """Check every watched artifact once. `dedup` maps
    `(artifact_id, source_id[, "moved"|"failed"])` -> last logged value, and
    is mutated in place so the caller owns its lifetime (a process restart
    re-announces, which is the honest behaviour -- we cannot know what was
    read).

    `checked` COUNTS RECORDED VERDICTS, not attempts. It is incremented after
    `record_update` returns, so an artifact whose verdict could not be
    written is not reported as checked -- the count would otherwise claim to
    have checked the one artifact nobody can see a result for.

    DEVIATION, ruled Important by the coordinator (task-6-report.md's second
    "Fix round"): two hardenings, both confirmed reachable at the time.

    1. `isinstance(entry, dict)` is checked before `entry.get("watch")` --
       `(entry or {}).get(...)` only substitutes `{}` for a falsy entry
       (`None`, `""`, `{}`); a truthy non-dict entry passes straight through
       and raises AttributeError, outside the per-artifact try below.

       WHAT MAKES IT REACHABLE TODAY IS THE STORE ARGUMENT, not the file.
       `store` is any object with `.get()`/`.record_update()`, and a
       hand-edited `provenance.json` no longer produces this shape:
       `ProvenanceStore._load()` rebuilds every entry through `_stored_entry`
       (app/provenance.py:181-228, Task 5 round 4). The guard stays because
       it defends this function's own contract, not the store's.

    2. The dispatch/record/log body for one artifact is wrapped in its own
       try/except. `dispatch()` guards every individual source, but
       `store.record_update` re-reads the file itself under its own lock,
       so a value that changes shape between this loop's snapshot and that
       fresh read (or any other future failure on this path) must not take
       a SIBLING artifact down with it -- the requirement this task exists
       to guarantee ("one failing source/artifact must not stop the pass")
       stated for the artifact loop, not just the source loop.
    """
    checked = 0
    available = 0

    for artifact_id, entry in sorted(store.get().items()):
        if not isinstance(entry, dict):
            continue                     # corrupt entry: nothing to watch, never crash
        sources = entry.get("watch") or []
        if not sources:
            continue                     # no origin, nothing to watch (U10)

        try:
            results = [dispatch(source, fetch) for source in sources]
            status = store.record_update(artifact_id, results)
            checked += 1
            if status == updates.AVAILABLE:
                available += 1
            for result in results:
                _log_transitions(events_path, artifact_id, result, dedup)
        except Exception as exc:  # noqa: BLE001 — per-artifact isolation, see docstring
            # Isolation, not silence [max-review c6]: one artifact's failure
            # must not end the pass, but a fully silent `continue` meant an
            # artifact could fail every pass forever with nothing anywhere to
            # show for it.
            #
            # A DISTINCT kind from the whole-pass `update-check-error`
            # (UpdateChecker's supervisor catch below), which carries
            # {"error"} only. Reusing it here with an artifact_id would be
            # one kind with two incompatible detail shapes — the vocabulary
            # defect this module already documents at that catch. `-error`
            # still ends in a suffix ui/src/model/eventSeverity.ts:27 reads
            # as failure severity, so the Events tab needs no new mapping.
            events.log_event(events_path, "update-check-artifact-error", {
                "artifact_id": artifact_id,
                "error": f"{type(exc).__name__}: {exc}"})
            continue

    return {"checked": checked, "available": available}


def _log_transitions(events_path, artifact_id, result, dedup) -> None:
    """DEVIATION FROM THE BRIEF, ruled on by the task coordinator after
    review (see task-6-report.md's "Fix round" addendum for the full
    writeup) -- two changes from what Step 3 originally shipped:

    1. A MOVED result logs `origin-moved` ONLY, never also
       `update-check-failed`. The two events describe the same fact at two
       levels of usefulness -- `origin-moved` is the specific diagnosis and
       the exact remedy (re-declare the remote); `update-check-failed` adds
       only "and therefore no verdict", which `origin-moved` already
       implies. Emitting both put two lines in the Events tab for one fact,
       with the less useful one indistinguishable from an ordinary network
       blip. The artifact still records UNAVAILABLE as its status --
       unchanged, this is purely about which event line fires.

    2. The `#failed` dedup key now clears whenever a source reports
       anything OTHER than UNAVAILABLE (not only when the brief's original
       `else` fired). Previously: fail, log once; recover; fail again with
       the identical note; permanently suppressed, because nothing ever
       cleared `#failed` on recovery. That makes a recurring or flapping
       failure -- exactly what an operator most needs to see -- invisible
       after its first occurrence, which is worse than the spam the dedup
       exists to prevent. Symmetric with how the AVAILABLE dedup key
       already clears on any non-available status.

       UNDETERMINED specifically: it is a successful read that could not be
       ranked, not a failure and not a pending "available". It clears
       `#failed` (via the same "anything other than UNAVAILABLE" rule) and
       falls through to the `else` branch below, which clears the AVAILABLE
       key too rather than holding it as pending -- it was never set for an
       UNDETERMINED result in the first place, since only the `if status ==
       AVAILABLE` branch ever writes `dedup[key]`.

    3. (Bundled Minor, same review round as 1/2 above) `#moved` now clears
       whenever a result does NOT report `detail.moved`, the same asymmetry
       fix ruling 2 already applied to `#failed`. As written it was only
       ever OVERWRITTEN when a new location differed from the stored one,
       never POPPED when a source stopped reporting moved -- so move to X,
       get re-declared/fixed, then move to X again would not re-log
       `origin-moved`, the identical "flapping becomes invisible" failure
       mode left asymmetric for this key.
    """
    # A TUPLE KEY, NOT A JOINED STRING. `f"{artifact_id}/{result['id']}"` had
    # a real collision surface: weights artifact ids legitimately contain "/"
    # (the relpath -- app/provenance_collect.py:110) and source ids are
    # free-form operator strings, so ("oci:local:a/b", "c") and
    # ("oci:local:a", "b/c") produced the same key and one artifact's event
    # silently suppressed the other's. A tuple removes the class rather than
    # picking a separator nobody can use.
    key = (artifact_id, result["id"])
    moved_key = (*key, "moved")
    failed_key = (*key, "failed")
    status = result["status"]
    moved = bool(result.get("detail", {}).get("moved"))

    if moved:
        location = result["detail"].get("location")
        if dedup.get(moved_key) != location:
            dedup[moved_key] = location
            events.log_event(events_path, "origin-moved", {
                "artifact_id": artifact_id, "source": result["id"],
                "location": location})
    else:
        dedup.pop(moved_key, None)

    if status != updates.UNAVAILABLE:
        # Any successful read -- CURRENT, AVAILABLE, or UNDETERMINED --
        # clears the remembered failure so a later recurrence of the exact
        # same note logs again instead of being silently swallowed by
        # stale dedup state from before the source recovered.
        dedup.pop(failed_key, None)

    if status == updates.AVAILABLE:
        if dedup.get(key) != result["latest"]:
            dedup[key] = result["latest"]
            events.log_event(events_path, "update-available", {
                "artifact_id": artifact_id, "source": result["id"],
                "current": result["current"], "latest": result["latest"],
                "detail": result["detail"]})
    elif status == updates.UNAVAILABLE:
        # A moved repository already got its one event above; a second,
        # more generic "no verdict" line for the identical fact is exactly
        # the double-logging ruling 1 removed.
        if not moved and result.get("note"):
            if dedup.get(failed_key) != result["note"]:
                dedup[failed_key] = result["note"]
                events.log_event(events_path, "update-check-failed", {
                    "artifact_id": artifact_id, "source": result["id"],
                    "note": result["note"]})
    else:
        dedup.pop(key, None)             # back to current/undetermined: re-announce next time


class UpdateChecker:
    """Slow-cadence upstream check on its OWN thread (arbiter.Watcher and
    storage.StorageWatcher idiom: stop event, daemon thread, ticks catch-all
    so the loop survives).

    It is a separate thread rather than another pass on the watcher tick for
    one blunt reason: that tick is synchronous and runs the reconciler, so a
    slow upstream would stall the machinery that keeps models loaded. The
    handoff is ProvenanceStore, which already locks and writes atomically.
    """

    def __init__(self, settings, provenance_store, events_path):
        self._settings = settings
        self._store = provenance_store
        self._events_path = events_path
        self._interval = getattr(settings, "update_interval_s", 21600.0)
        self._enabled = getattr(settings, "update_check_enabled", True)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dedup: dict = {}
        self._etags: dict = {}
        # ONE PASS AT A TIME. POST /api/provenance/check runs tick() on the
        # request thread (app/routers/provenance.py:210) while this object's
        # own thread may already be mid-pass. They share `_dedup` and
        # `_etags` with no synchronisation of their own -- concurrent passes
        # can double-log the same transition -- and each builds a FRESH
        # `fetch` closure, so the 40-request budget is per PASS. Two
        # overlapping passes spend it twice against GitHub's 60/hour
        # anonymous per-IP ceiling, which app/updates/fetch.py:8-14 names as
        # the reason the budget exists at all.
        self._tick_lock = threading.Lock()
        # DEVIATION FROM THE BRIEF: app/updates/fetch.py's own module
        # docstring explicitly directs this thread to build ONE httpx.Client
        # and pass it into every pass's `make_fetch(client=...)` call rather
        # than let `make_fetch` open a fresh default client every tick, purely
        # to reuse TCP/TLS connections across the six-hour cadence. The
        # brief's `_fetch_factory` builds a fresh client every call instead
        # (`make_fetch(etags=self._etags)`, no `client=`), which contradicts
        # that guidance from the very module it depends on. Never explicitly
        # closed -- fetch.py's docstring establishes plain refcounting is
        # fine for this, and here the client's lifetime matches the
        # checker's own anyway, which is the whole point of holding it.
        self._client = httpx.Client(follow_redirects=False)
        self._fetch_factory = lambda: make_fetch(etags=self._etags, client=self._client)

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run,
                                        name="model-deck-update-checker",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self._interval, 10.0) + 5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            if self._stop.wait(self._interval):
                break

    def tick(self) -> dict | None:
        if not self._enabled or self._store is None:
            return None
        with self._tick_lock:            # see _tick_lock's comment in __init__
            try:
                fetch = self._fetch_factory()
                return run_pass(self._store, fetch, self._events_path,
                                dedup=self._dedup)
            except Exception as exc:  # noqa: BLE001 — supervisor loop, never silent
                # `update-check-error`, NOT `update-check-failed`: that kind
                # is the per-source verdict and carries
                # {"artifact_id","source","note"} (:245-248, _log_transitions).
                # One kind with two incompatible detail shapes is the exact
                # vocabulary defect this codebase has already shipped four
                # times. Both end in a suffix ui/src/model/eventSeverity.ts
                # classifies as a failure, so the Events tab is unchanged.
                events.log_event(self._events_path, "update-check-error",
                                 {"error": f"{type(exc).__name__}: {exc}"})
                return None
