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

Since sglang-omni Task 6 this module also holds the per-(node, RESOURCE)
half of the same idea — `RemoteEngineClients` for engines DECLARED on a
node-agent entry, the cross-registry walk that finds them, that node's GPU
pool, and `remote_world_half`, the ONE assembly of the remote half of a
world snapshot that both the arbiter tick and the HTTP paths call. Same
posture throughout: None means "not operable right now", never an
exception into a tick; no engine kind is named anywhere (per-kind
constructor knowledge stays on app.engine_kinds' adapters).

Locking mirrors NodeStore's: one non-reentrant lock around the whole
check-compare-rebuild body, so two threads racing a rotation cannot both
build a client (the loser would leak an unclosed transport). Registry reads
inside the lock are NodeStore reads, which are lock-free by design — no
lock-ordering hazard exists.
"""

from __future__ import annotations

import threading

import httpx

from app.engine_kinds import ENGINE_KINDS
from app.engines import EngineError
from app.observe import SparkObserver

# The agent reports VRAM in MiB (node-agent models.IndividualGPU's
# memory_*_mb, filled from nvidia-smi/rocm-smi); the deck's world speaks
# bytes with a derived `free` (app.gpu.read_gpus). ONE translation, in
# `read_remote_gpus` below — a second copy is how the next units bug gets in.
_MIB = 1024**2


def binding_view(store, entry: dict) -> dict:
    """The three fields a swap client is bound from, as the registry holds
    them now. The credential rides as a digest
    (node_store.credential_fingerprint), so a rotation is detectable without
    the value ever reaching anything wire-facing. (Moved verbatim from
    app/node_binding.py, deleted by N1 T12 once every reader migrated here.)"""
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
            try:
                client = self._factory(entry, self._store.credential_for(node_id))
            except (ValueError, httpx.InvalidURL):
                # A hand-edited row the write gate never saw (node_store's
                # _usable_url refuses these on the wire): not operable is
                # the honest answer, and returning None keeps a bad address
                # from crashing the arbiter tick that asked. Narrow catch —
                # construction failures only, never probe errors.
                return None
            self._built[node_id] = (view, client)
            return client

    def _retire(self, node_id: str) -> None:
        built = self._built.pop(node_id, None)
        if built is not None:
            built[1].close()

    def retire_absent(self, keep_ids) -> None:
        """Close and drop built clients whose ids are not in `keep_ids` —
        the demotion/removal path, where nothing will ever ask client_for
        for the id again, so retire-on-read can never fire. Called from
        NodeObservers.snapshot()'s own retirement branch; lock order is
        always Observers -> Clients (NodeClients never calls back into
        NodeObservers), so the nested acquisition cannot invert."""
        with self._lock:
            for node_id in list(self._built):
                if node_id not in keep_ids:
                    self._retire(node_id)


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
            # Retire the CLIENT too — see retire_absent's docstring: once an
            # observer is gone nothing will call client_for(node_id) again,
            # so retire-on-read can never fire and the old client would leak
            # otherwise.
            self._clients.retire_absent(swap_ids)
            return dict(self._observers)

    def observer_for(self, node_id: str):
        return self.snapshot().get(node_id)

    def invalidate(self, node_id: str) -> None:
        """Drop one node's observation cache — call after acting on it (a
        swap), whose whole purpose is to change what the cache holds."""
        observer = self.snapshot().get(node_id)
        if observer is not None:
            observer.invalidate()


# ===========================================================================
# Remote DECLARED engines (sglang-omni Task 6) — the per-resource
# counterpart of the per-node machinery above.
#
# A node-agent entry may now carry an `engines[]` declaration of its own
# (Task 5), so the deck needs three things the swap path never needed: the
# cross-registry walk of those declarations, a client PER (node, resource)
# rather than per node, and that node's GPU pool. All three keep
# NodeClients' posture exactly — not operable is a STATE (None), never an
# exception raised into a reconciler tick — and none of them names an
# engine kind (spec §8): the per-kind constructor knowledge lives on the
# adapters, same as app.local_clients delegates to `build_client`.
# ===========================================================================


def remote_engine_declarations(node_store) -> list[dict]:
    """Every engine DECLARED on a registry entry other than the local one,
    each stamped with its owning ``node_id``.

    The stamp is the point: an `engines[]` entry says nothing about which
    box it is on, and every key downstream (observation, intent, the world
    map) is built from `(node_id, resource)`. Skipped by ``agent_kind ==
    "local"`` rather than by id, because that is the field NodeStore
    actually constrains (`id` "local" is reserved FOR that agent_kind, not
    the other way round).

    Re-reads the registry on every call — the NodeObservers precedent: a
    declaration edit applies live, no restart.
    """
    declared: list[dict] = []
    for entry in node_store.list():
        if entry["agent_kind"] == "local":
            continue
        for engine in entry.get("engines") or []:
            declared.append({**engine, "node_id": entry["id"]})
    return declared


def read_remote_gpus(node_store, agent_client_factory,
                     node_ids) -> dict[str, list[dict] | None]:
    """Each named node's GPU pool, in the deck's own world shape
    (``{index, total, used, free}``, bytes), read through the node-agent's
    existing ``GET /v1/node/gpu``.

    ``None`` for a node that could not be read AT ALL — no address, no
    stored credential, the agent did not answer, or it answered with a body
    that isn't the documented shape. That single value is what the caller
    turns into "this node's engines are unknown": the gpu probe alone
    governs a node's liveness, exactly as app/node_observer.py already binds
    a node's whole status to it.

    Never raises. A powered-off box is the NORMAL state of a swap node, and
    a malformed body is a bug on the other machine — neither may take down
    the arbiter tick or the /api/state request that asked (the N1 lesson:
    malformed/unreachable HEALS, it never kills the tick).
    """
    pools: dict[str, list[dict] | None] = {}
    for node_id in sorted(node_ids):
        entry = node_store.get(node_id)
        if (entry is None
                or not entry.get("address")
                or not node_store.credential_set(node_id)):
            # "Not set up" and "not answering" both read as "we could not
            # look" here; app/node_observer.py is where that distinction is
            # surfaced to the operator, and it stays its job.
            pools[node_id] = None
            continue
        try:
            client = agent_client_factory(entry["address"],
                                          node_store.credential_for(node_id))
        except (ValueError, httpx.InvalidURL):
            # A hand-edited address the write gate never saw — same narrow
            # construction-only catch NodeClients.client_for makes, for the
            # same reason.
            pools[node_id] = None
            continue
        try:
            payload = client.gpu()
        except EngineError:
            pools[node_id] = None
            continue
        finally:
            client.close()
        pools[node_id] = _gpu_pool(payload)
    return pools


def _gpu_pool(payload) -> list[dict] | None:
    """The agent's gpu payload -> the deck's world gpu shape, or None if it
    isn't that shape at all. Narrow catch, boundary-only: this is another
    machine's response body, not our own data structure."""
    try:
        return [
            {"index": int(gpu["index"]),
             "total": int(gpu["memory_total_mb"]) * _MIB,
             "used": int(gpu["memory_used_mb"]) * _MIB,
             "free": (int(gpu["memory_total_mb"]) - int(gpu["memory_used_mb"])) * _MIB}
            for gpu in payload["gpus"] or []
        ]
    except (KeyError, TypeError, ValueError):
        return None


def _adapter_remote_client(entry: dict, credential: str, engine: dict):
    """The default `RemoteEngineClients` factory: build this engine's client
    through its KIND's own remote constructor (app.engine_kinds), the exact
    delegation app.local_clients makes to ``build_client`` for local ones,
    so no engine name appears here.

    ``build_remote_client`` is OPTIONAL on the adapter protocol. A kind that
    has none cannot run off-box at all — None ("not operable") is the honest
    answer for a declaration that names it, not a crash. Every kind is in
    that state today; Task 7 ships the first one that isn't.
    """
    build = getattr(ENGINE_KINDS[engine["kind"]], "build_remote_client", None)
    if build is None:
        return None
    return build(entry["address"], credential, engine["resource"],
                 engine["connection"])


class RemoteEngineClients:
    """Lazy, self-healing map of (node id, resource) -> remote engine client.

    NodeClients' pattern per RESOURCE instead of per node: the binding view
    is that node's `binding_view` (address/serving_address/credential
    fingerprint) PLUS the declared engine's own `(kind, connection)`, so an
    address rotation, a credential rotation and a connection edit each
    rebuild, and the retired client is closed rather than leaked.

    `client_for` answers None — never raises — for any pair that is not
    operable right now: unknown node, the local node, a node-agent entry
    with no address or no stored credential (design §9: a credential can
    vanish mid-flight), a resource that node doesn't declare, or a kind with
    no remote constructor. It runs inside reconciler ticks and HTTP handlers
    alike, where "not operable" is a state, not an error.
    """

    def __init__(self, node_store, client_factory=None):
        self._store = node_store
        # (entry, credential, engine) -> client | None. Injected so tests
        # never open sockets; the default delegates to the engine kind.
        self._factory = client_factory or _adapter_remote_client
        self._lock = threading.Lock()
        # (node_id, resource) -> (binding view, client) for clients built.
        self._built: dict[tuple[str, str], tuple[tuple, object]] = {}

    def client_for(self, node_id: str, resource: str):
        with self._lock:
            entry = self._store.get(node_id)
            if (entry is None
                    or entry.get("agent_kind") != "node-agent"
                    or not entry.get("address")
                    or not self._store.credential_set(node_id)):
                self._retire((node_id, resource))
                return None
            engine = next((e for e in entry.get("engines") or []
                           if e["resource"] == resource), None)
            if engine is None:
                self._retire((node_id, resource))
                return None
            view = (binding_view(self._store, entry), engine["kind"],
                    sorted(engine["connection"].items()))
            built = self._built.get((node_id, resource))
            if built is not None and built[0] == view:
                return built[1]
            self._retire((node_id, resource))
            try:
                client = self._factory(entry, self._store.credential_for(node_id),
                                       engine)
            except (ValueError, httpx.InvalidURL):
                # Construction failures only, never probe errors — see
                # NodeClients.client_for's own comment.
                return None
            if client is None:
                return None
            self._built[(node_id, resource)] = (view, client)
            return client

    def _retire(self, pair: tuple[str, str]) -> None:
        built = self._built.pop(pair, None)
        if built is not None:
            built[1].close()

    def retire_absent(self, keep_pairs) -> None:
        """Close and drop built clients whose (node, resource) pair is not
        in `keep_pairs` — the undeclared/removed path, where nothing will
        ever ask client_for for that pair again, so retire-on-read can never
        fire. Mirrors NodeClients' method of the same name."""
        with self._lock:
            for pair in list(self._built):
                if pair not in keep_pairs:
                    self._retire(pair)


def remote_world_half(node_store, agent_client_factory, remote_clients,
                      world, registry) -> dict:
    """The REMOTE half of a world snapshot: ``{"remote_gpus",
    "remote_tenants"}``, ready to merge into what ``World.snapshot``
    returned for the local half.

    ONE function, called from BOTH ``app.routers.build_world_snapshot`` and
    ``app.arbiter.Watcher.tick`` — the two callers must agree about the
    world, and they share ONE ``World`` instance whose idle-clock memory is
    pruned against whatever declaration each call passes it. Two
    hand-rolled copies of this assembly would let the HTTP surface and the
    arbiter declare different things and thrash each other's clocks; that
    is the same reason ``build_world_snapshot``'s docstring gives for
    re-snapshotting through the shared instance rather than caching.

    Either dependency absent (every unit test, any caller built before this
    task, a deck with the watcher disabled) means no remote half at all —
    empty, not a crash.
    """
    if node_store is None or remote_clients is None or agent_client_factory is None:
        return {"remote_gpus": {}, "remote_tenants": {}}
    engines = remote_engine_declarations(node_store)
    # Only nodes that actually DECLARE something are probed: a registered
    # node-agent the deck merely watches is app/node_observer.py's business,
    # and probing it from here would spend a transport timeout per tick for
    # nothing.
    pools = read_remote_gpus(node_store, agent_client_factory,
                             {e["node_id"] for e in engines})
    remote_clients.retire_absent({(e["node_id"], e["resource"]) for e in engines})
    return {
        "remote_gpus": pools,
        "remote_tenants": world.snapshot_remote(engines, remote_clients, pools,
                                                registry),
    }
