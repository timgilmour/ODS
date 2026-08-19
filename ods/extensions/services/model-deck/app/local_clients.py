"""Per-resource actuation/observation clients for DECLARED local engines,
the LocalClients counterpart to app.node_clients.NodeClients (swap nodes).

Mirrors NodeClients' lazy, self-healing pattern exactly (see that module's
docstring): a client is built on first use from the CURRENT declaration
entry's connection, and rebuilt (old one dropped, never closed — none of
today's engine client classes expose a close(), so there is nothing new to
leak here that _build_deck's one-shot construction didn't already leave
unclosed) whenever the entry's `(kind, connection)` pair changes, or
dropped entirely when the resource disappears from the declaration.
`client_for` answers None — never raises — for a resource that isn't
declared, exactly NodeClients' "not operable right now is a state, not an
error" posture.

Construction dispatches to `app.engine_kinds.ENGINE_KINDS[kind].build_client(
connection, settings, node_store)` (review fix, T3 round 2: this module used to hold
the per-kind constructor dispatch itself — a disclosed engine-kind-name
residue — moved onto the adapters instead, since per-kind constructor
knowledge is exactly what app.engine_kinds exists to hold; see that
module's docstring for hipfire's DockerCtl/LiteLLMClient-from-Settings
note). This module now names no engine kind anywhere.
"""

from __future__ import annotations

import threading

from app.engine_kinds import ENGINE_KINDS


def _build_key(entry: dict) -> tuple:
    return (entry["kind"], frozenset(entry["connection"].items()))


class LocalClients:
    """Lazy, self-healing map of resource -> declared local engine client
    (design §3 / Task 3 brief; mirrors app.node_clients.NodeClients)."""

    def __init__(self, node_store, settings):
        self._store = node_store
        self._settings = settings
        self._lock = threading.Lock()
        # resource -> (build_key, client) for clients actually built.
        self._built: dict[str, tuple[tuple, object]] = {}

    def client_for(self, resource: str):
        """The current client for `resource`, or None when it is not
        declared right now. Rebuilds when the declaration's (kind,
        connection) pair changed since the last build."""
        with self._lock:
            local = self._store.get("local")
            entry = None
            if local is not None:
                for e in local.get("engines", []):
                    if e["resource"] == resource:
                        entry = e
                        break
            if entry is None:
                self._retire(resource)
                return None
            key = _build_key(entry)
            built = self._built.get(resource)
            if built is not None and built[0] == key:
                return built[1]
            self._retire(resource)
            # Not guarded: NodeStore validates `entry["kind"]` against
            # engine_kinds.KNOWN_KINDS (engine_kinds.validate_engines)
            # before an entry can ever land in the declaration this reads
            # from — an unknown kind here is a real bug, not a
            # user-reachable state, so a bare KeyError is the correct
            # "let it crash" signal (matches World.snapshot's own
            # ENGINE_KINDS[entry["kind"]] lookup, app/state.py).
            client = ENGINE_KINDS[entry["kind"]].build_client(
                entry["connection"], self._settings, self._store)
            self._built[resource] = (key, client)
            return client

    def _retire(self, resource: str) -> None:
        self._built.pop(resource, None)

    def retire_absent(self, keep_resources) -> None:
        """Drop built clients whose resource is not in `keep_resources` —
        called after a declaration edit removes/renames a resource, so a
        stale client is never handed out again. Mirrors NodeClients'
        method of the same name (app/node_clients.py)."""
        with self._lock:
            for resource in list(self._built):
                if resource not in keep_resources:
                    self._retire(resource)
