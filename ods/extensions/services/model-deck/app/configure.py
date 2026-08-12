"""How settings reach an engine — dispatch over the declared mech.

The write boundary is NOT "local vs remote". It is what each engine's
capability descriptor declares:

* ``api``          — lemonade: a live call, no reload. DECLARED, NOT BUILT.
* ``env+restart``  — hipfire, comfyui: write env, reload later. DECLARED,
  NOT BUILT.
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

``api`` and ``env+restart`` are listed above because ``MECHS`` names them,
but NO engine client implements ``configure()`` or ``set_env()`` — they
were never built. ``apply_settings`` refuses them explicitly rather than
dispatching into an AttributeError [max-review c1/c12]. Today's only live
caller (app.routers.spark) uses ``node-settings``.
"""

from app.engines import EngineError

MECHS = ("api", "env+restart", "node-settings", "none")


class UnbuiltMechError(EngineError):
    """A MECHS name whose dispatch was never built (see the refusal at the
    bottom of apply_settings). Subclasses EngineError so every non-HTTP
    caller's ``except EngineError`` treatment is unchanged; the app-wide
    handler renders it 501 Not Implemented instead of EngineError's 502 —
    502 says "the engine is broken", which sends an operator debugging the
    wrong side of the wire [T9 review m-item, 2026-08-10]."""


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

    # Both remaining mechs are named in MECHS but UNBUILT: no engine
    # client in app/engines implements configure() or set_env() (verified by
    # grep across the package — only test fakes do). Dispatching would raise
    # AttributeError deep inside a route, which reads as a deck bug rather
    # than as "this was never built" [max-review c1/c12]. Refuse in the
    # contract's own vocabulary instead; build the mech before restoring the
    # dispatch. (MECHS is the only place they are named — no capability
    # descriptor anywhere declares either one [T9 review m8].)
    #
    # Placed AFTER the empty-values check above deliberately: applying
    # nothing stays a no-op rather than becoming an error, which is what
    # keeps "a save with no settings must not wipe an engine's config" true.
    raise UnbuiltMechError(
        f"configure mech {mech!r} is declared but not implemented by any engine client")
