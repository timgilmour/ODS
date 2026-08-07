"""How settings reach an engine — dispatch over the declared mech.

The write boundary is NOT "local vs remote". It is what each engine's
capability descriptor declares:

* ``api``          — lemonade: a live call, no reload.
* ``env+restart``  — hipfire, comfyui: write env, reload later.
* ``node-settings``— spark and future remote engines: ship a settings
  document to the node-agent (Plan C2 Task 7), whose host-side helper merges
  it into argv at the next swap. Ships and returns ``requires_reload`` —
  applying it is always the human's Reload click (see app.routers.spark),
  never this call.
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


def apply_settings(
    mech: str,
    *,
    engine_client,
    resolved: dict,
    profile: str | None = None,
    env: dict | None = None,
    argv: list | None = None,
    service: str | None = None,
) -> dict:
    """Apply `resolved` settings via `mech`. Returns an outcome record.

    `profile`/`env`/`argv`/`service` are node-settings-only (Plan C2): every
    other mech ignores them, matching each engine's own configure()/set_env()
    call shape, which has no notion of a profile or a launch argv."""
    if mech not in MECHS:
        raise ValueError(f"unknown configure mech {mech!r}; expected one of {MECHS}")

    if mech == "node-settings":
        if not profile:
            raise ValueError("node-settings requires a profile")
        # The one shape node-agent's settings_store accepts, verbatim (see
        # node-agent/settings_store.py's EMPTY/_KEYS) — args here is
        # WHATEVER `resolved` was handed, declared-only filtering already
        # done by the caller (app.routers.spark.spark_reload), not this
        # function's concern.
        document = {
            "args": {key: entry["value"] for key, entry in resolved.items()},
            "env": dict(env or {}),
            "argv": list(argv or []),
            "service": service,
        }
        engine_client.put_settings(profile, document)
        return {"applied": True, "requires_reload": True,
                "reason": "settings shipped to the node; reload to apply"}

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
