"""``plan_rename`` — the alias -> identity rename migration PLANNER (Plan
C2, Task 10).

Naming rule this planner enforces (ontology design, binding): a model's
IDENTITY is its checkpoint directory name, verbatim — never a served alias.
Sparky's vLLM profiles were adopted (Task 5) before that rule existed, so
several `--served-model-name` lists carry role-flavoured aliases (``aeon``,
``aeon-fast``, ``aeon-deep``, ...) that all serve ONE checkpoint. This
module plans the migration from "several aliases" to "one identity, plus
tags for the role-flavoured ones" — it does not perform it. Renaming a
profile's served name, regenerating litellm routes, and telling a pinned
client (OMP) to repoint are all real, order-sensitive actions with their
own failure modes; bundling them into a "planner" would make this function
impure and its output untrustworthy. Execution is deliberately a future
plan with its own gate.

PURE, by design: no I/O (no store reads, no engine calls — the caller
gathers `profiles`/`routes` and passes them in), and never mutates any of
its three arguments — a caller may hand this the SAME dict twice, from a
route two requests apart, and get back the same answer both times, which
is exactly what makes `POST /api/rename/plan` safe to call any number of
times.

Two decisions worth being explicit about, since neither is exercised by the
"exact" test (`test_multi_alias_profile_collapses_to_one_identity`), which
only has one un-suffixed alias per profile:

* **Which alias is "the primary"?** Given a profile's served-name list, the
  FIRST alias that carries no recognized role suffix is treated as this
  profile's un-flavoured/default name — it is folded into the identity
  silently (no tag, no note): that is the whole POINT of a multi-alias
  profile collapsing to one identity. Any role-suffixed alias becomes a
  `proposed_tags` entry instead. A SECOND (or later) un-suffixed alias in
  the same list is genuinely ambiguous — this fleet's naming convention
  doesn't produce one, but silently dropping it would hide a real fact —
  so it is reported as a `notes` entry rather than guessed at.
* **Duplicate tags.** mm27b's real compose ships two `-ultimate` aliases
  (`aeon-ultimate` and `qwen36-ultimate`) — `proposed_tags` is a proposed
  TAG SET to apply to the renamed identity, not one entry per alias, so
  duplicates collapse (first-seen order preserved) rather than doubling up.

Collisions (two profiles' `identity` agreeing) are reported, never picked —
the whole point of an ontology violation like this is that a human decides
which profile is stale, not this function.
"""

# Role-flavoured suffixes this fleet's naming convention actually uses.
# Checked in this order only for docstring/readability -- match is by exact
# suffix, so order never changes which one wins (a name can only end with
# one of these at a time).
_ROLE_SUFFIXES = ("fast", "deep", "ultimate", "xs")


def _classify_dropped_aliases(served: list, identity: str) -> tuple[list, list]:
    """Split one profile's served-name list (everything except a value that
    already equals `identity`) into role-flavoured `proposed_tags` and a
    `notes` list for anything else this module has no rule for -- see the
    module docstring, "Which alias is 'the primary'?". Alias order is
    preserved for tags; a role suffix is only recorded once even if more
    than one alias carries it (see the module docstring, "Duplicate
    tags")."""
    tags: list = []
    notes: list = []
    seen_primary = False

    for alias in served:
        if alias == identity:
            continue

        matched_suffix = next(
            (suffix for suffix in _ROLE_SUFFIXES if alias.endswith(f"-{suffix}")),
            None,
        )
        if matched_suffix is not None:
            if matched_suffix not in tags:
                tags.append(matched_suffix)
            continue

        if not seen_primary:
            seen_primary = True
            continue

        notes.append(alias)

    return tags, notes


def plan_rename(profiles: dict, routes: dict, client_pins: dict) -> dict:
    """Plan the alias -> identity migration. See the module docstring for
    the full rule set; the short version:

    * `profiles`: `{<profile>: {"served_model_name": [...], "identity":
      str}}`. A profile whose served-name list is already exactly
      `[identity]` needs no rename and produces no entry.
    * `routes`: the flat litellm `route_table()` shape, `{<route_name>:
      <litellm_params.model>}` -- values may carry the "openai/" prefix
      real `/model/info` responses use (app.engines.litellm module
      docstring); this function strips it before joining, mirroring
      `app.routers.facts._gateway_runtime`'s rule: NEVER `model_name`
      (that is the route ALIAS), always the resolved `litellm_params.model`.
    * `client_pins`: `{<route_name>: [<pin description>, ...]}`, caller
      supplied -- this deck cannot see OMP's (or any other client's)
      config, so a caller (the route, ultimately a human reading the
      runbook) names the known pins itself. Every pin the caller supplies
      is surfaced in the output; this function cannot verify or refute one.

    Returns `{"renames": [...], "collisions": [...], "client_pins": [...]}`.
    Two profiles resolving to the SAME identity is a collision: reported,
    and NEITHER profile gets a rename entry -- see the module docstring.
    """
    by_identity: dict = {}
    for profile, meta in profiles.items():
        by_identity.setdefault(meta["identity"], []).append(profile)

    collisions = [
        {"identity": identity, "profiles": sorted(owners)}
        for identity, owners in by_identity.items()
        if len(owners) > 1
    ]
    collided_identities = {c["identity"] for c in collisions}

    renames = []
    for profile, meta in profiles.items():
        identity = meta["identity"]
        if identity in collided_identities:
            continue

        served = list(meta["served_model_name"])
        if served == [identity]:
            continue

        proposed_tags, notes = _classify_dropped_aliases(served, identity)
        entry = {
            "profile": profile,
            "from": served,
            "to": identity,
            "proposed_tags": proposed_tags,
        }
        if notes:
            entry["notes"] = notes
        renames.append(entry)

    renamed_aliases = {alias for rename in renames for alias in rename["from"]}

    plan_pins = []
    seen_routes = set()
    for route_name, resolved_model in routes.items():
        target = resolved_model.removeprefix("openai/")
        if target in renamed_aliases:
            plan_pins.append({
                "route": route_name,
                "targets": target,
                "pins": list(client_pins.get(route_name, [])),
            })
            seen_routes.add(route_name)

    for route_name, pins in client_pins.items():
        if route_name in seen_routes:
            continue
        plan_pins.append({"route": route_name, "pins": list(pins)})

    return {"renames": renames, "collisions": collisions, "client_pins": plan_pins}
