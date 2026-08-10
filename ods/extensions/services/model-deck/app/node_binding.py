"""Is the deck's ACTUATION path still bound to current node configuration?

The asymmetry this module reports [max-review #13]: OBSERVATION re-reads the
registry every tick (app.node_observer), while ACTUATION binds a SparkClient
exactly once, at app build (app.main, where it also stashes what it bound as
``deck["spark_bound"]``). So editing sparky's address on the Nodes screen
moves monitoring immediately and leaves swaps/restores pointing at the boot
address until a restart. That consequence was documented in main.py's comment
and surfaced to the operator nowhere.

Lives here, not in a router, because TWO surfaces must answer it identically:
``/api/state``'s node block (app.routers.status._nodes_block — what the Nodes
screen actually renders) and the registry CRUD list (app.routers.nodes). Two
copies of this rule would drift, and the one the UI reads is the one that
matters.

Pure: no app, no request, no I/O beyond the store read in ``binding_view``.
"""

from app.observe import spark_node_id


def actuation_stale(bound: dict | None, current: dict) -> bool:
    """``bound`` is what app.main captured when it built the SparkClient
    (``None`` if it built none); ``current`` is the same three fields as the
    registry holds them now.

    ``bound is None`` + a configured registry means "a restart would now bind
    a client, and today's deck cannot actuate this node at all" — stale for
    the same operator-facing reason. Unbound AND unconfigured is not stale:
    nothing is out of date, so there is nothing to nag about.
    """
    if bound is None:
        # Mirrors app.main's bind condition (address + serving_address +
        # credential). Anything less would not produce a client on restart
        # either, so it is not yet a pending rebind.
        return bool(current["address"] and current["serving_address"]
                    and current["credential_fp"])
    return bound != current


def binding_view(store, entry: dict) -> dict:
    """The three fields app.main binds a SparkClient from, as the registry
    holds them now. Same keys as main.py's ``spark_bound`` stash — these two
    dicts are compared directly, so their shapes must not drift apart.

    The credential rides as a digest (node_store.credential_fingerprint), so
    a rotation is detectable without this value ever reaching an API whose
    credential contract is write-only.
    """
    return {
        "address": entry.get("address"),
        "serving_address": entry.get("serving_address"),
        "credential_fp": store.credential_fingerprint(entry["id"]),
    }


def entry_actuation_stale(store, entry: dict, bound: dict | None) -> bool:
    """The whole rule for one registry entry, as both surfaces report it.

    Only the spark node can be stale: it is the one resource the deck
    actuates through a bound client. Every other entry is observe-only, so it
    has no binding to go stale — reported ``False`` rather than omitted,
    because a field present on some rows and absent on others makes every
    consumer write an existence check.
    """
    if entry["id"] != spark_node_id():
        return False
    return actuation_stale(bound, binding_view(store, entry))
