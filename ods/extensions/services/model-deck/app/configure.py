"""How settings reach an engine — dispatch over the declared mech.

The write boundary is NOT "local vs remote". It is what each engine's
capability descriptor declares:

* ``api``          — lemonade: a live call, no reload.
* ``env+restart``  — hipfire, comfyui: write env, reload later.
* ``node-settings``— spark and future remote engines: ship a settings
  document to the node-agent, whose host-side helper merges it into argv at
  launch. **Implemented in Plan C2**, and raising here rather than stubbing:
  a stub that reported success while dropping the settings would be far
  worse than an error that says what it is.
* ``none``         — a source you don't own (someone else's API): read and
  warn, permanently. This slot exists so the general rule stays honest.

**Nothing here restarts anything.** A save changes intent; the reload that
applies launch-class settings is always a human click (Tim, 2026-08-04).
``env+restart`` therefore writes the environment and reports
``requires_reload``; it does not act.

Applying an EMPTY settings map is a no-op, never a wipe — "I have nothing to
say about this engine" must not mean "clear its configuration".
"""

MECHS = ("api", "env+restart", "node-settings", "none")


def apply_settings(mech: str, *, engine_client, resolved: dict) -> dict:
    """Apply `resolved` settings via `mech`. Returns an outcome record.

    No caller in C1 (settings are savable, but "apply"/reload is a later
    increment — this function is wired up when reload lands)."""
    if mech not in MECHS:
        raise ValueError(f"unknown configure mech {mech!r}; expected one of {MECHS}")

    if mech == "node-settings":
        raise NotImplementedError(
            "configure.mech 'node-settings' lands in Plan C2 (node-agent settings "
            "endpoint + swap-helper argv merge)"
        )

    if mech == "none":
        return {
            "applied": False,
            "requires_reload": False,
            "reason": "this source cannot be configured by the Deck (read and warn only)",
        }

    values = {key: entry["value"] for key, entry in resolved.items()}
    if not values:
        return {"applied": False, "requires_reload": False, "reason": "no settings to apply"}

    if mech == "api":
        engine_client.configure(values)
        return {"applied": True, "requires_reload": False, "reason": "applied live"}

    # env+restart
    engine_client.set_env(values)
    return {
        "applied": True,
        "requires_reload": True,
        "reason": "environment updated; reload to apply",
    }
