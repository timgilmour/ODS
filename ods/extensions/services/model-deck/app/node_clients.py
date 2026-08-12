"""Per-node actuation clients and observers, bound lazily and rebound live.

Replaces app.main's one-shot boot bind and app/node_binding.py's staleness
REPORTING [max-review #13]: observation used to re-read the registry every
tick while actuation bound a SparkClient exactly once at app build, so an
address edit moved monitoring immediately and left swaps/restores pointing
at the boot address until a restart. Every actuation path now takes its
client from HERE, per call, so there is no boot-time binding to go stale —
the asymmetry is deleted, not surfaced.

`client_for` is repair-shaped, never wire-shaped: it answers None (caller
vocabulary: `unconfigured`, app/node_observer.py's word) rather than
raising, because it runs inside reconciler ticks and HTTP handlers alike
and "this node is not operable right now" is a state, not an error.

Locking mirrors NodeStore's: one non-reentrant lock around the whole
check-compare-rebuild body, so two threads racing a rotation cannot both
build a client (the loser would leak an unclosed transport). Registry reads
inside the lock are NodeStore reads, which are lock-free by design — no
lock-ordering hazard exists.
"""

from __future__ import annotations

import threading

from app.observe import SparkObserver


def binding_view(store, entry: dict) -> dict:
    """The three fields a swap client is bound from, as the registry holds
    them now. The credential rides as a digest
    (node_store.credential_fingerprint), so a rotation is detectable without
    the value ever reaching anything wire-facing. (Moved verbatim from
    app/node_binding.py, which N1 deletes.)"""
    return {
        "address": entry.get("address"),
        "serving_address": entry.get("serving_address"),
        "credential_fp": store.credential_fingerprint(entry["id"]),
    }


class NodeClients:
    """Lazy, self-healing map of node id -> actuation client (design §3)."""

    def __init__(self, node_store, client_factory):
        self._store = node_store
        # (entry, credential) -> client. Injected so tests never open
        # sockets and app.main decides the real class exactly once.
        self._factory = client_factory
        self._lock = threading.Lock()
        # node_id -> (binding_view, client) for clients actually built.
        self._built: dict[str, tuple[dict, object]] = {}

    def client_for(self, node_id: str):
        """The current client for `node_id`, or None when the node is not
        operable (missing, control != "swap", or a prerequisite absent —
        e.g. its credential vanished mid-flight, design §9). Rebuilds when
        the binding view changed; the old client is closed, not leaked."""
        with self._lock:
            entry = self._store.get(node_id)
            if (entry is None
                    or entry.get("control") != "swap"
                    or not entry.get("address")
                    or not entry.get("serving_address")
                    or not self._store.credential_set(node_id)):
                self._retire(node_id)
                return None
            view = binding_view(self._store, entry)
            built = self._built.get(node_id)
            if built is not None and built[0] == view:
                return built[1]
            self._retire(node_id)
            client = self._factory(entry, self._store.credential_for(node_id))
            self._built[node_id] = (view, client)
            return client

    def _retire(self, node_id: str) -> None:
        built = self._built.pop(node_id, None)
        if built is not None:
            built[1].close()


class NodeObservers:
    """One SparkObserver per control:"swap" node, created and retired as the
    registry changes (the same lazy pattern as NodeClients; observer
    TTL/backoff semantics unchanged — design §5).

    Each observer's spark_fn resolves through NodeClients.client_for at
    probe time, so an observer never holds a client: a rebind or demotion
    is picked up on the very next probe. Retirement drops the observer's
    backoff state with it — a re-promoted node starts fresh, which is
    correct (its old failure history described a different declaration).
    """

    def __init__(self, node_store, node_clients: NodeClients,
                 observer_factory=None):
        self._store = node_store
        self._clients = node_clients
        # (spark_fn, node_id) -> observer. node_id rides along for tests;
        # the real SparkObserver ignores it.
        self._factory = observer_factory or (
            lambda fn, node_id: SparkObserver(fn))
        self._lock = threading.Lock()
        self._observers: dict[str, object] = {}

    def snapshot(self) -> dict[str, object]:
        """The live observer map. Re-reads the registry on every call (the
        node_observer precedent: add/remove/credential changes apply live,
        no restart); the store read is one small-file JSON load."""
        with self._lock:
            swap_ids = {n["id"] for n in self._store.list()
                        if n.get("control") == "swap"}
            for node_id in swap_ids:
                if node_id not in self._observers:
                    self._observers[node_id] = self._factory(
                        lambda nid=node_id: self._clients.client_for(nid),
                        node_id)
            for node_id in list(self._observers):
                if node_id not in swap_ids:
                    del self._observers[node_id]
            return dict(self._observers)

    def observer_for(self, node_id: str):
        return self.snapshot().get(node_id)

    def invalidate(self, node_id: str) -> None:
        """Drop one node's observation cache — call after acting on it (a
        swap), whose whole purpose is to change what the cache holds."""
        observer = self.snapshot().get(node_id)
        if observer is not None:
            observer.invalidate()
