"""Model Deck status derivation — intent x observed -> one status.

The bug this exists to kill: a deliberately parked engine and a dead one
produce *identical observations*. ods-hipfire sat Exited(0) for 26 hours on
2026-08-03 while the Deck displayed it exactly as it displays a park, and
nothing anywhere said otherwise. Status therefore cannot be read off an
observation alone — it is a function of the observation AND what we
intended.

Pure functions only, no I/O, following the repo's functional-core
convention (see app/state.py). The caller supplies the intent record
(app.intent) and the observation (app.observe).

Ordering rules that are easy to get wrong:

* **Reachability is checked first.** An unreachable node retains its
  last-known intent and reports ``unreachable`` — never ``down``. The
  ontology's ``unavailable != empty`` rule, one level up from storage.
* **Quarantine is checked after a healthy match.** A quarantined resource
  that is nonetheless serving must report ``serving``: quarantine describes
  our restore attempts, not reality, and displaying "quarantined" over a
  live model would be a lie an operator would (rightly) stop trusting.
"""

STATUSES = (
    "serving",      # intent loaded, observed loaded, same model
    "drifted",      # intent loaded, observed loaded, DIFFERENT model
    "down",         # intent loaded, observed not loaded on a reachable node
    "parked",       # intent unloaded, observed not loaded — deliberate
    "unexpected",   # intent unloaded, observed loaded — someone else acted
    "unmanaged",    # no intent, observed loaded — adopt candidate
    "idle",         # no intent, nothing loaded
    "unreachable",  # node did not answer; last-known intent retained
    "quarantined",  # restore attempts exhausted; awaiting an operator
    "warming",      # a load/boot is in flight — transient, never actionable
)


def derive_status(intent: dict | None, observed: dict) -> dict:
    """Return ``{"status", "reason"}`` for one resource.

    ``intent`` is an app.intent record or None (nothing ever recorded).
    ``observed`` is ``{"reachable": bool, "loaded": bool, "model": str|None}``.
    """
    if not observed["reachable"]:
        return {"status": "unreachable", "reason": "node did not answer; last-known state retained"}

    if observed.get("transitioning"):
        # A boot in flight is neither loaded nor dead. Checked before
        # everything except reachability so a slow start can never be read
        # as a failure and restarted into a storm.
        return {"status": "warming", "reason": "a load or boot is in flight"}

    loaded = observed["loaded"]
    actual = observed["model"]

    if intent is None:
        if loaded:
            return {"status": "unmanaged", "reason": f"{actual!r} is loaded but the Deck has no intent for it"}
        return {"status": "idle", "reason": "no intent recorded and nothing loaded"}

    wanted = intent["model"]

    if intent["state"] == "loaded":
        # wanted is None means "loaded, no opinion which model" — the correct
        # reading for single-model engines like hipfire, whose model the Deck
        # does not choose. Treating None as a name to match would report
        # permanent drift for a perfectly healthy engine.
        if loaded and (wanted is None or actual == wanted):
            return {"status": "serving", "reason": f"{actual!r} serving as intended"}
        if loaded:
            return {
                "status": "drifted",
                "reason": f"intended {wanted!r} but {actual!r} is loaded",
            }
        if intent.get("quarantined"):
            return {
                "status": "quarantined",
                "reason": f"restore of {wanted!r} failed repeatedly; awaiting operator",
            }
        return {"status": "down", "reason": f"intended {wanted!r} is not loaded"}

    # intent["state"] == "unloaded" — a deliberate park.
    if loaded:
        return {
            "status": "unexpected",
            "reason": f"deliberately unloaded, but {actual!r} is loaded",
        }
    return {"status": "parked", "reason": "deliberately unloaded"}
