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

from app import events, updates
from app.updates import git as git_checks
from app.updates import oci as oci_checks

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
    "artifact_id/source_id" -> last logged `latest`, and is mutated in place
    so the caller owns its lifetime (a process restart re-announces, which is
    the honest behaviour -- we cannot know what was read)."""
    checked = 0
    available = 0

    for artifact_id, entry in sorted(store.get().items()):
        sources = (entry or {}).get("watch") or []
        if not sources:
            continue                     # no origin, nothing to watch (U10)
        checked += 1

        results = [dispatch(source, fetch) for source in sources]
        status = store.record_update(artifact_id, results)
        if status == updates.AVAILABLE:
            available += 1

        for result in results:
            _log_transitions(events_path, artifact_id, result, dedup)

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
    """
    key = f"{artifact_id}/{result['id']}"
    status = result["status"]
    moved = bool(result.get("detail", {}).get("moved"))

    if moved:
        location = result["detail"].get("location")
        if dedup.get(f"{key}#moved") != location:
            dedup[f"{key}#moved"] = location
            events.log_event(events_path, "origin-moved", {
                "artifact_id": artifact_id, "source": result["id"],
                "location": location})

    if status != updates.UNAVAILABLE:
        # Any successful read -- CURRENT, AVAILABLE, or UNDETERMINED --
        # clears the remembered failure so a later recurrence of the exact
        # same note logs again instead of being silently swallowed by
        # stale dedup state from before the source recovered.
        dedup.pop(f"{key}#failed", None)

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
            if dedup.get(f"{key}#failed") != result["note"]:
                dedup[f"{key}#failed"] = result["note"]
                events.log_event(events_path, "update-check-failed", {
                    "artifact_id": artifact_id, "source": result["id"],
                    "note": result["note"]})
    else:
        dedup.pop(key, None)             # back to current/undetermined: re-announce next time
