"""Model Deck reconciler — which statuses justify acting, and which never do.

Pure decision function in the repo's functional-core style (mirrors
``app.arbiter.decide``): it returns actions, it does not perform them. The
Watcher executes and records outcomes.

Exactly ONE status produces an action: ``down`` — intent says loaded, the
node is reachable, nothing is loaded. Every other status is deliberately
inert, and each refusal is a real incident rather than caution for its own
sake:

* ``parked`` — restoring it would fight a deliberate unload every tick.
  This is the single most important invariant in the lifecycle work.
* ``drifted`` / ``unexpected`` / ``unmanaged`` — the Deck did not author
  this state, so it must report and let a human decide. Auto-correcting
  someone else's action is how a supervisor becomes an adversary.
* ``quarantined`` — the failure budget is spent; retrying is the crash
  loop we are here to prevent.
* ``unreachable`` — a node being off is not a model having fallen over.

Two global suppressions: ``auto_enabled`` (the policy toggle) and
``boot_window_active`` (right after a node boots, "not loaded yet" and
"died" are indistinguishable; waiting costs seconds, guessing wrong costs
a multi-minute swap).
"""

ACTIONABLE_STATUS = "down"


def plan_reconcile(
    statuses: dict[str, dict],
    intents: dict[str, dict],
    *,
    auto_enabled: bool,
    boot_window_active: bool,
) -> list[dict]:
    """Return the restore actions justified by `statuses`, in key order."""
    if not auto_enabled or boot_window_active:
        return []

    actions = []
    for key in sorted(statuses):
        if statuses[key]["status"] != ACTIONABLE_STATUS:
            continue
        intent = intents.get(key)
        if intent is None:
            # A 'down' status is only derivable from an intent record, so
            # this means intent was dropped between derivation and planning
            # (a concurrent forget). Skip rather than invent one.
            continue
        actions.append({
            "action": "restore",
            "key": key,
            "engine": intent["engine"],
            "model": intent["model"],
            "reason": "intent is loaded but nothing is loaded",
        })
    return actions
