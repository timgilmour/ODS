"""Engine-kind registry — THE one module that knows local engine names.

E1 (spec §2) ships this as an internal protocol with exactly three kinds;
E2 turns it into the pluggable descriptor registry. Spec §8 binds: no
engine name may appear in app/ outside this module (allowed residues are
listed in the plan's Global Constraints).

This task ships only the declaration half: kind names + connection
schemas + validate_engines. Adapters (observe/verbs/idle/reclaim) land in
Task 3."""

# kind -> {connection field -> required?}
KNOWN_KINDS: dict[str, dict[str, bool]] = {
    "lemonade": {"url": True, "metrics_url": True, "container": True},
    "comfyui": {"url": True},
    "hipfire": {"container": True},
}

_POLICY_FIELDS = {"priority": int, "pinned": bool, "idle_ttl": int}


def _bad(reason: str) -> ValueError:
    return ValueError(reason)


def validate_engines(engines: object) -> None:
    """Raise ValueError (one-line reason) unless `engines` is a valid
    declaration list. Refuse, never coerce ([[literal-declared-inputs]])."""
    if not isinstance(engines, list):
        raise _bad("engines must be a list")
    seen: set[str] = set()
    for e in engines:
        if not isinstance(e, dict):
            raise _bad("engine entry must be an object")
        extra = set(e) - {"resource", "kind", "connection", "gpu_index",
                          "policy_defaults"}
        if extra:
            raise _bad(f"engine entry has extra field(s): {sorted(extra)}")
        resource = e.get("resource")
        if (not isinstance(resource, str) or not resource
                or "/" in resource or resource != resource.strip()):
            raise _bad("resource must be a non-empty string without '/'")
        if resource in seen:
            raise _bad(f"duplicate resource {resource!r}")
        seen.add(resource)
        kind = e.get("kind")
        if kind not in KNOWN_KINDS:
            raise _bad(f"unknown kind {kind!r} (known: {sorted(KNOWN_KINDS)})")
        schema = KNOWN_KINDS[kind]
        conn = e.get("connection")
        if not isinstance(conn, dict):
            raise _bad(f"{resource}: connection must be an object")
        extra_conn = set(conn) - set(schema)
        if extra_conn:
            raise _bad(f"{resource}: connection has extra field(s): "
                       f"{sorted(extra_conn)}")
        for field, required in schema.items():
            if required and not (isinstance(conn.get(field), str) and conn[field]):
                raise _bad(f"{resource}: connection.{field} is required "
                           f"for kind {kind!r}")
        gpu = e.get("gpu_index")
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise _bad(f"{resource}: gpu_index must be a non-negative integer")
        pol = e.get("policy_defaults")
        if not isinstance(pol, dict) or set(pol) != set(_POLICY_FIELDS):
            raise _bad(f"{resource}: policy_defaults must have exactly "
                       f"{sorted(_POLICY_FIELDS)}")
        for field, typ in _POLICY_FIELDS.items():
            v = pol[field]
            if typ is int and (isinstance(v, bool) or not isinstance(v, int)):
                raise _bad(f"{resource}: policy_defaults.{field} must be int")
            if typ is bool and not isinstance(v, bool):
                raise _bad(f"{resource}: policy_defaults.{field} must be bool")
