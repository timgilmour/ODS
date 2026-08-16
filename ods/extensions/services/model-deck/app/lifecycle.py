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


def join_warming(observed: dict[str, dict], world: dict,
                 intents: dict[str, dict], now=None) -> dict[str, dict]:
    """Mark every observation a kind considers a BOOT IN FLIGHT as
    ``transitioning``, so ``derive_status`` reads it as ``warming`` rather
    than ``down`` (sglang-omni Task 9).

    THIS IS THE SEAM, and its placement is the whole point. A kind whose
    "not serving yet" and "died" observations are byte-identical needs the
    INTENT record's timestamp to tell them apart (app.engine_kinds'
    ``_SglangOmniAdapter.warming``, GF4: a cold start takes ~3.5-4.5
    minutes). app.state/app.observe cannot supply that — they are
    intent-blind BY DESIGN, which is the separation THIS module exists to
    bridge — so the join happens here, where intent and observation already
    meet, and both consumers of that pair call it: the arbiter's reconcile
    pass (which must not restore-storm a booting engine into quarantine) and
    ``app.routers.build_lifecycle_view`` (so the board says the same word the
    reconciler is acting on).

    `world` is the raw snapshot, whose tenants carry each kind's OWN
    vocabulary — `warming` is written against that, not against the record
    shape below, because it is the ADAPTER's rule about its own engine. The
    two are joined by key here rather than by re-expressing one in the
    other's terms.

    `warming` is OPTIONAL on the adapter protocol: a kind that does not
    define it has no boot window (the E1 triple — lemonade's in-flight load
    and hipfire's container-up-not-healthy are already sourced as
    `transitioning` in app.observe, from the OBSERVATION alone, needing no
    intent), and this function leaves its observations untouched.

    Returns a NEW mapping (records copied): the caller's own `observed` stays
    exactly what app.observe produced, which keeps this a pure stage in the
    functional core this module belongs to.
    """
    from app.engine_kinds import ENGINE_KINDS
    from app.observe import local_key, node_key

    # Keys built from each tenant's OWN fields, the same way app.observe's
    # two halves build theirs — never re-derived from a map key, so the
    # world map and the observation map cannot drift apart here either.
    keyed = [(local_key(resource), tenant)
             for resource, tenant in (world.get("tenants") or {}).items()]
    keyed += [(node_key(tenant["node_id"], tenant["resource"]), tenant)
              for tenant in (world.get("remote_tenants") or {}).values()]

    joined = {key: dict(record) for key, record in observed.items()}
    for key, tenant in keyed:
        warming = getattr(ENGINE_KINDS[tenant["engine"]], "warming", None)
        record = joined.get(key)
        if warming is None or record is None:
            continue
        if warming(intents.get(key), tenant, now):
            record["transitioning"] = True
    return joined
