"""NodeClients/NodeObservers — the lazy self-healing rebind (design §3).

Fixtures use boxa/boxb, labels ≠ ids, plus one control:"none" node WITH a
serving_address ([[defaults-that-hide-bugs]]): operability must come from
the declaration alone, never inferred from data presence.
"""

import httpx
import pytest

from app.engines.node_agent import NodeAgentUnreachable
from app.node_clients import (
    NodeClients,
    NodeObservers,
    RemoteEngineClients,
    read_remote_gpus,
    remote_engine_declarations,
    remote_world_half,
)
from app.node_store import NodeStore
from app.state import World


class FakeClient:
    def __init__(self, entry, credential):
        self.entry = dict(entry)
        self.credential = credential
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def store(tmp_path):
    s = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")
    s.add({"id": "boxa", "label": "Box Alpha", "agent_kind": "node-agent",
           "address": "http://boxa:7720", "serving_address": "http://boxa:8000",
           "control": "swap"}, credential="key-boxa")
    s.add({"id": "boxb", "label": "Box Beta", "agent_kind": "node-agent",
           "address": "http://boxb:7720", "serving_address": "http://boxb:8000",
           "control": "swap"}, credential="key-boxb")
    # Has everything EXCEPT the declaration — must never get a client.
    s.add({"id": "watcher", "label": "Watch Only", "agent_kind": "node-agent",
           "address": "http://watcher:7720",
           "serving_address": "http://watcher:8000"}, credential="key-watcher")
    return s


@pytest.fixture
def clients(store):
    return NodeClients(store, FakeClient)


def test_client_for_operable_node(clients):
    client = clients.client_for("boxa")
    assert client is not None
    assert client.credential == "key-boxa"
    assert client.entry["address"] == "http://boxa:7720"


def test_control_none_gets_no_client_despite_serving_address(clients):
    assert clients.client_for("watcher") is None


def test_unknown_node_gets_none(clients):
    assert clients.client_for("ghost") is None


def test_same_view_returns_same_client(clients):
    assert clients.client_for("boxa") is clients.client_for("boxa")


@pytest.mark.parametrize("mutate", [
    lambda s: s.update("boxa", {"address": "http://boxa2:7720"}),
    lambda s: s.update("boxa", {"serving_address": "http://boxa2:8000"}),
    lambda s: s.update("boxa", {"label": "Renamed"}, credential="rotated-key"),
])
def test_rebuilds_on_each_binding_view_change(store, clients, mutate):
    first = clients.client_for("boxa")
    mutate(store)
    second = clients.client_for("boxa")
    assert second is not first
    assert first.closed          # old client retired, not leaked


def test_label_change_does_not_rebuild(store, clients):
    first = clients.client_for("boxa")
    store.update("boxa", {"label": "Box Alpha Prime"})
    assert clients.client_for("boxa") is first
    assert not first.closed


def test_demotion_retires_and_returns_none(store, clients):
    first = clients.client_for("boxa")
    store.update("boxa", {"control": "none"})
    assert clients.client_for("boxa") is None
    assert first.closed


def test_two_nodes_two_clients(clients):
    a, b = clients.client_for("boxa"), clients.client_for("boxb")
    assert a is not None and b is not None and a is not b
    assert (a.credential, b.credential) == ("key-boxa", "key-boxb")


# --- N1 T8 review: a factory construction failure heals to None, not a raise
# (a hand-edited row the write-side gate in app.node_store never saw -- its
# usable-URL check is refused on the wire now, but this is defense at the
# repair boundary too: httpx.Client(base_url=...) can still raise
# httpx.InvalidURL, which is NOT a ValueError subclass, so both must be
# caught explicitly).


def test_client_for_heals_to_none_when_factory_raises_value_error(store):
    def factory(entry, credential):
        raise ValueError("bad url")

    clients = NodeClients(store, factory)

    assert clients.client_for("boxa") is None


def test_client_for_heals_to_none_when_factory_raises_httpx_invalid_url(store):
    def factory(entry, credential):
        raise httpx.InvalidURL("bad url")

    clients = NodeClients(store, factory)

    assert clients.client_for("boxa") is None


def test_client_for_construction_failure_does_not_poison_the_cache(store):
    """First attempt raises (simulating a bad row); nothing is cached, so a
    later call with a working factory result still succeeds -- proves the
    failed build never got stored in `_built`."""
    calls = {"n": 0}

    def factory(entry, credential):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad url")
        return FakeClient(entry, credential)

    clients = NodeClients(store, factory)

    assert clients.client_for("boxa") is None       # first attempt: raises
    second = clients.client_for("boxa")              # retried, not poisoned
    assert second is not None
    assert calls["n"] == 2


# --- NodeObservers -----------------------------------------------------------

def test_observers_one_per_swap_node(store, clients):
    observers = NodeObservers(store, clients)
    snap = observers.snapshot()
    assert set(snap) == {"boxa", "boxb"}       # never "watcher"


def test_observers_retire_on_demotion(store, clients):
    observers = NodeObservers(store, clients)
    assert "boxa" in observers.snapshot()
    store.update("boxa", {"control": "none"})
    assert set(observers.snapshot()) == {"boxb"}


def test_snapshot_retires_the_client_on_demotion_without_client_for(store, clients):
    """Demotion/removal never calls client_for(id) again — nothing else in
    the deck would have a reason to. Before retire_absent, the built client
    stayed open until process exit; now NodeObservers.snapshot()'s own
    retirement branch closes it too, with no further client_for call."""
    observers = NodeObservers(store, clients)
    observers.snapshot()                       # builds boxa's observer
    built = clients.client_for("boxa")          # the client under test
    store.update("boxa", {"control": "none"})
    observers.snapshot()                        # demotion path only
    assert built.closed


def test_observer_status_none_when_client_unbindable(store, clients):
    """The observer's spark_fn goes through client_for, so a node whose
    credential vanished mid-flight reads None -> observe_spark emits no key
    (design §9)."""
    observers = NodeObservers(store, clients)
    store.update("boxa", {"control": "none"})
    observer = observers.snapshot().get("boxa")
    assert observer is None       # retired outright


def test_invalidate_reaches_the_right_observer(store, clients):
    calls = []

    class FakeObserver:
        def __init__(self, node_id):
            self.node_id = node_id
        def invalidate(self):
            calls.append(self.node_id)

    observers = NodeObservers(store, clients,
                              observer_factory=lambda fn, nid: FakeObserver(nid))
    observers.snapshot()
    observers.invalidate("boxb")
    assert calls == ["boxb"]


# ===========================================================================
# sglang-omni Task 6 — RemoteEngineClients + the remote half of the world.
#
# Fixture discipline ([[defaults-that-hide-bugs]]): node "nimbus" (NOT
# "sparky"), resource "gguf-r" (NOT "omni"), resource != kind name,
# gpu_index 4. The registry states here are HAND-BUILT (conftest's
# HandBuiltRegistry) because the Task 5 write gate refuses a LEMONADE-kind
# declaration on a node-agent entry — still true after Task 7, which made
# only sglang-omni remote-capable — and these tests deliberately use one, so
# nothing here can pass by riding on the one kind the paths under test were
# built for. See that class's docstring; the gate is not weakened.
# ===========================================================================

_R_CONNECTION = {"url": "http://gguf-r:8080",
                 "metrics_url": "http://gguf-r:8081/metrics",
                 "container": "gguf-r"}

_R_ENGINE = {"resource": "gguf-r", "kind": "lemonade",
             "connection": _R_CONNECTION, "gpu_index": 4,
             "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}}


def _entries():
    return [
        {"id": "local", "label": "This Box", "agent_kind": "local",
         "control": "none",
         "engines": [{"resource": "gguf-a", "kind": "lemonade",
                      "connection": {"url": "http://gguf-a:8080",
                                     "metrics_url": "http://gguf-a:8081/metrics",
                                     "container": "gguf-a"},
                      "gpu_index": 2,
                      "policy_defaults": {"priority": 50, "pinned": False,
                                          "idle_ttl": 900}}]},
        {"id": "nimbus", "label": "Nimbus Box", "agent_kind": "node-agent",
         "address": "http://nimbus:7720", "control": "none",
         "engines": [dict(_R_ENGINE)]},
        # Reachable, credentialled, declares NOTHING — must never produce a
        # declaration, a client, or a GPU probe.
        {"id": "bystander", "label": "Bystander", "agent_kind": "node-agent",
         "address": "http://bystander:7720", "control": "none", "engines": []},
    ]


@pytest.fixture
def registry(hand_built_registry):
    return hand_built_registry(_entries(),
                               {"nimbus": "key-nimbus", "bystander": "key-by"})


class _FakeAgent:
    """NodeAgentClient-shaped: `gpu()` + `close()` only (the two the remote
    GPU read actually calls)."""

    def __init__(self, payload=None, raises=None) -> None:
        self.payload = payload
        self.raises = raises
        self.closed = False
        self.calls = 0

    def gpu(self) -> dict:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.payload

    def close(self) -> None:
        self.closed = True


def _agent_factory(agent):
    """One recording factory returning the SAME agent for every address."""
    opened = []

    def factory(address, credential):
        opened.append((address, credential))
        return agent

    factory.opened = opened
    return factory


class _RemoteEngineClient:
    def __init__(self, entry, credential, engine) -> None:
        self.entry = dict(entry)
        self.credential = credential
        self.engine = dict(engine)
        self.closed = False

    def close(self) -> None:
        self.closed = True


# --- the declaration walk ---------------------------------------------------


def test_remote_engine_declarations_skips_the_local_entry(registry):
    declared = remote_engine_declarations(registry)

    assert [e["resource"] for e in declared] == ["gguf-r"]


def test_remote_engine_declarations_stamp_the_owning_node_id(registry):
    """The engine entry alone doesn't say which box it is on; everything
    downstream keys off this stamp."""
    declared = remote_engine_declarations(registry)

    assert declared[0]["node_id"] == "nimbus"
    assert declared[0]["kind"] == "lemonade"
    assert declared[0]["gpu_index"] == 4


def test_remote_engine_declarations_on_an_empty_registry_is_empty(
        hand_built_registry):
    assert remote_engine_declarations(hand_built_registry([])) == []


# --- the per-node GPU pool --------------------------------------------------


def test_read_remote_gpus_translates_the_agent_payload_into_the_decks_shape(
        registry):
    """The agent reports MiB (models.IndividualGPU.memory_*_mb); the deck's
    world speaks bytes with a derived `free`. One translation, here."""
    agent = _FakeAgent({"gpus": [{"index": 4, "memory_total_mb": 8,
                                  "memory_used_mb": 3}]})

    pools = read_remote_gpus(registry, _agent_factory(agent), {"nimbus"})

    assert pools == {"nimbus": [{"index": 4, "total": 8 * 1024**2,
                                 "used": 3 * 1024**2,
                                 "free": 5 * 1024**2}]}


def test_read_remote_gpus_is_none_when_the_agent_does_not_answer(registry):
    agent = _FakeAgent(raises=NodeAgentUnreachable("connection refused"))

    assert read_remote_gpus(registry, _agent_factory(agent),
                            {"nimbus"}) == {"nimbus": None}


def test_read_remote_gpus_is_none_when_the_node_has_no_credential(registry):
    """"Not set up" and "not answering" must not collapse into a crash —
    both read as "we could not look", and neither is probed."""
    registry.set_credential("nimbus", None)
    agent = _FakeAgent({"gpus": []})

    pools = read_remote_gpus(registry, _agent_factory(agent), {"nimbus"})

    assert pools == {"nimbus": None}
    assert agent.calls == 0


def test_read_remote_gpus_is_none_on_a_malformed_payload(registry):
    """Malformed heals, never kills the tick (the N1 lesson): a body the
    agent should never send must not take the arbiter down."""
    agent = _FakeAgent({"gpus": [{"index": 4}]})   # no memory fields

    assert read_remote_gpus(registry, _agent_factory(agent),
                            {"nimbus"}) == {"nimbus": None}


def test_read_remote_gpus_closes_the_client_it_opened(registry):
    agent = _FakeAgent({"gpus": []})

    read_remote_gpus(registry, _agent_factory(agent), {"nimbus"})

    assert agent.closed is True


def test_read_remote_gpus_never_raises_on_a_hand_edited_address(registry):
    registry.patch("nimbus", address="not a url")

    def factory(address, credential):
        raise httpx.InvalidURL(address)

    assert read_remote_gpus(registry, factory, {"nimbus"}) == {"nimbus": None}


# --- the lazy, self-healing client map --------------------------------------


@pytest.fixture
def remote_clients(registry):
    return RemoteEngineClients(registry, _RemoteEngineClient)


def test_remote_client_for_a_declared_engine(remote_clients):
    client = remote_clients.client_for("nimbus", "gguf-r")

    assert client.credential == "key-nimbus"
    assert client.entry["address"] == "http://nimbus:7720"
    assert client.engine["connection"] == _R_CONNECTION


def test_remote_client_for_same_view_returns_the_same_client(remote_clients):
    assert (remote_clients.client_for("nimbus", "gguf-r")
            is remote_clients.client_for("nimbus", "gguf-r"))


@pytest.mark.parametrize("mutate", [
    lambda r: r.patch("nimbus", address="http://nimbus2:7720"),
    lambda r: r.set_credential("nimbus", "rotated"),
    lambda r: r.patch("nimbus", engines=[{**_R_ENGINE,
                                          "connection": {**_R_CONNECTION,
                                                         "url": "http://moved:8080"}}]),
])
def test_remote_client_rebuilds_when_its_binding_changed(
        registry, remote_clients, mutate):
    first = remote_clients.client_for("nimbus", "gguf-r")

    mutate(registry)
    second = remote_clients.client_for("nimbus", "gguf-r")

    assert second is not first
    assert first.closed is True


@pytest.mark.parametrize("node_id, resource", [
    ("ghost", "gguf-r"),        # no such node
    ("local", "gguf-a"),        # the local entry is never remote
    ("bystander", "gguf-r"),    # node-agent, but declares nothing
    ("nimbus", "gguf-x"),       # node exists, resource is not declared on it
])
def test_remote_client_for_a_non_operable_pair_is_none(
        remote_clients, node_id, resource):
    assert remote_clients.client_for(node_id, resource) is None


def test_remote_client_for_is_none_once_the_credential_vanishes(
        registry, remote_clients):
    """design §9: a credential deleted out of band mid-flight. Not operable
    is a STATE, never an exception raised into a reconciler tick."""
    built = remote_clients.client_for("nimbus", "gguf-r")
    registry.set_credential("nimbus", None)

    assert remote_clients.client_for("nimbus", "gguf-r") is None
    assert built.closed is True


def test_remote_client_for_never_raises_on_a_hand_edited_address(registry):
    def factory(entry, credential, engine):
        raise httpx.InvalidURL(entry["address"])

    clients = RemoteEngineClients(registry, factory)

    assert clients.client_for("nimbus", "gguf-r") is None


def test_remote_client_for_is_none_when_the_kind_has_no_remote_constructor(
        registry):
    """The DEFAULT factory dispatches to the engine kind's own remote
    constructor. A kind that has none cannot run off-box at all — None
    ("not operable") is the honest answer, not a crash, and no engine name
    appears in the dispatch."""
    clients = RemoteEngineClients(registry)

    assert clients.client_for("nimbus", "gguf-r") is None


def test_remote_retire_absent_closes_dropped_clients(remote_clients):
    built = remote_clients.client_for("nimbus", "gguf-r")

    remote_clients.retire_absent({("nimbus", "gguf-z")})

    assert built.closed is True


# --- the assembled remote half ----------------------------------------------


def test_remote_world_half_is_empty_without_the_wiring(registry):
    """Every pre-Task-6 caller (unit tests, a watcher built without the
    remote deps) gets an empty remote half rather than a crash."""
    assert remote_world_half(registry, None, None, object(), object()) == {
        "remote_gpus": {}, "remote_tenants": {}}


def test_remote_world_half_probes_only_nodes_that_declare_engines(registry):
    agent = _FakeAgent({"gpus": []})
    factory = _agent_factory(agent)

    remote_world_half(registry, factory, RemoteEngineClients(registry),
                      World(), object())

    assert factory.opened == [("http://nimbus:7720", "key-nimbus")]


# ===========================================================================
# sglang-omni Task 7 — probe pacing (RemoteObserver).
#
# With the first `remote_capable` kind live, the remote half's probes are on
# the hot path: the per-node `GET /v1/node/gpu` and every declared engine's
# status fire on EVERY ~2 s arbiter tick AND every /api/state, each behind a
# 5 s transport timeout. One powered-off node would stretch every tick past
# 5 s — the exact problem SparkObserver already solves for the swap half
# (app/observe.py's own comments), so this mirrors its shape: a short TTL on
# success, a growing per-node backoff on failure.
#
# Fixture discipline: nodes "nimbus"/"cirrus" (never the live-seeded
# "sparky"), resource "song-r" (never "omni"), GPU 4, TTL/backoff set to
# values that are NOT the defaults so nothing passes by coincidence.
# ===========================================================================

_OMNI_ENGINE = {"resource": "song-r", "kind": "sglang-omni",
                "connection": {"url": "http://127.0.0.1:8008"}, "gpu_index": 4,
                "policy_defaults": {"priority": 5, "pinned": False,
                                    "idle_ttl": 120}}

_PACED_TTL_S = 7.0
_PACED_BACKOFF_S = 21.0


def _paced_entries():
    return [
        {"id": "local", "label": "This Box", "agent_kind": "local",
         "control": "none", "engines": []},
        {"id": "nimbus", "label": "Nimbus Box", "agent_kind": "node-agent",
         "address": "http://nimbus:7720", "control": "none",
         "engines": [dict(_OMNI_ENGINE)]},
        {"id": "cirrus", "label": "Cirrus Box", "agent_kind": "node-agent",
         "address": "http://cirrus:7720", "control": "none",
         "engines": [dict(_OMNI_ENGINE)]},
    ]


class _CountingOmni:
    """A remote engine client that counts its probes — the per-tenant half
    of "at most one probe per node per TTL"."""

    def __init__(self) -> None:
        self.status_calls = 0

    def status(self) -> dict:
        self.status_calls += 1
        return {"reachable": True, "healthy": True, "busy_requests": 0}

    def close(self) -> None:
        pass


class _Paced:
    """One RemoteObserver plus the fakes it paces, on a fake clock."""

    def __init__(self, registry, agents, ttl_s=_PACED_TTL_S,
                 backoff_base_s=_PACED_BACKOFF_S, backoff_max_s=600.0) -> None:
        from app.node_clients import RemoteObserver

        self.registry = registry
        self.agents = agents            # node id -> _FakeAgent
        self.opened = []
        self.engines = {}               # (node, resource) -> _CountingOmni
        self.now = 1000.0
        self.world = World()
        self.observer = RemoteObserver(ttl_s=ttl_s, backoff_base_s=backoff_base_s,
                                       backoff_max_s=backoff_max_s,
                                       clock=lambda: self.now)
        self.remote_clients = RemoteEngineClients(registry, self._engine_client)

    def _engine_client(self, entry, credential, engine):
        key = (entry["id"], engine["resource"])
        return self.engines.setdefault(key, _CountingOmni())

    def _agent_factory(self, address, credential):
        self.opened.append(address)
        for node_id, agent in self.agents.items():
            if address == f"http://{node_id}:7720":
                return agent
        raise AssertionError(f"unexpected address {address!r}")

    def half(self) -> dict:
        return self.observer.half(self.registry, self._agent_factory,
                                  self.remote_clients, self.world, object())

    def gpu_probes(self, node_id) -> int:
        return self.agents[node_id].calls

    def status_probes(self, node_id) -> int:
        client = self.engines.get((node_id, "song-r"))
        return 0 if client is None else client.status_calls


def _pool(index=4):
    return {"gpus": [{"index": index, "memory_total_mb": 8, "memory_used_mb": 3}]}


@pytest.fixture
def paced(hand_built_registry):
    registry = hand_built_registry(
        _paced_entries(), {"nimbus": "key-nimbus", "cirrus": "key-cirrus"})
    return _Paced(registry, {"nimbus": _FakeAgent(_pool()),
                             "cirrus": _FakeAgent(_pool())})


def test_remote_observer_probes_each_node_at_most_once_within_the_ttl(paced):
    """THE pacing property: N ticks inside one TTL cost one probe per node —
    both halves, the agent's GPU read AND each declared engine's status."""
    for _ in range(5):
        paced.now += 1.0            # five ~ticks, all inside the 7 s TTL
        paced.half()

    assert paced.gpu_probes("nimbus") == 1
    assert paced.gpu_probes("cirrus") == 1
    assert paced.status_probes("nimbus") == 1
    assert paced.status_probes("cirrus") == 1


def test_remote_observer_serves_the_same_observation_from_the_cache(paced):
    """Not merely fewer probes — the same ANSWER, so a tick and an
    /api/state in the same second can never describe different worlds."""
    first = paced.half()
    paced.now += 1.0
    second = paced.half()

    assert second["remote_tenants"]["nimbus/song-r"]["state"] == "idle"
    assert second == first


def test_remote_observer_reprobes_once_the_ttl_lapses(paced):
    paced.half()

    paced.now += _PACED_TTL_S
    paced.half()

    assert paced.gpu_probes("nimbus") == 2
    assert paced.status_probes("nimbus") == 2


def test_remote_observer_serves_the_cached_unknown_after_a_failure(paced):
    """A powered-off box is the NORMAL state of a remote node. The first
    probe eats one transport timeout; every tick until the TTL lapses is
    served from cache — unknown, never "nothing is loaded"."""
    paced.agents["nimbus"] = _FakeAgent(raises=NodeAgentUnreachable("refused"))

    first = paced.half()
    for _ in range(4):
        paced.now += 1.0
        paced.half()

    assert paced.gpu_probes("nimbus") == 1
    assert first["remote_gpus"]["nimbus"] is None
    assert first["remote_tenants"]["nimbus/song-r"]["state"] == "unknown"
    # The engine itself was never probed: with the node's own liveness read
    # already failed, N per-engine probes would only buy N more timeouts.
    assert paced.status_probes("nimbus") == 0


def test_remote_observer_backs_a_failed_node_off_past_the_plain_ttl(paced):
    """A node that did not answer is not retried on the very next TTL — at
    a 10 s TTL a powered-off box would still cost a 5 s timeout every 10 s,
    which is the whole problem. Growing backoff, SparkObserver's shape."""
    paced.agents["nimbus"] = _FakeAgent(raises=NodeAgentUnreachable("refused"))
    paced.half()

    paced.now += _PACED_TTL_S            # TTL lapsed, backoff has not
    paced.half()

    assert paced.gpu_probes("nimbus") == 1
    assert paced.half()["remote_tenants"]["nimbus/song-r"]["state"] == "unknown"


def test_remote_observer_retries_a_failed_node_once_its_backoff_lapses(paced):
    paced.agents["nimbus"] = _FakeAgent(raises=NodeAgentUnreachable("refused"))
    paced.half()

    paced.now += _PACED_BACKOFF_S
    paced.half()

    assert paced.gpu_probes("nimbus") == 2


def test_remote_observer_backoff_grows_with_repeated_failures(paced):
    """Two failures must wait longer than one — otherwise a box that has
    been off for a week is polled as hard as one that blinked."""
    paced.agents["nimbus"] = _FakeAgent(raises=NodeAgentUnreachable("refused"))
    paced.half()
    paced.now += _PACED_BACKOFF_S
    paced.half()                          # second failure

    paced.now += _PACED_BACKOFF_S         # enough for the FIRST backoff only
    paced.half()

    assert paced.gpu_probes("nimbus") == 2


def test_remote_observer_backoff_is_capped(paced):
    """Uncapped doubling would eventually mean a node that recovered is
    never looked at again."""
    from app.node_clients import RemoteObserver

    paced.observer = RemoteObserver(ttl_s=_PACED_TTL_S, backoff_base_s=_PACED_BACKOFF_S,
                                    backoff_max_s=_PACED_BACKOFF_S,
                                    clock=lambda: paced.now)
    paced.agents["nimbus"] = _FakeAgent(raises=NodeAgentUnreachable("refused"))
    for _ in range(4):
        paced.half()
        paced.now += _PACED_BACKOFF_S

    assert paced.gpu_probes("nimbus") == 4


def test_remote_observer_keeps_a_live_node_fresh_while_a_dead_one_backs_off(paced):
    """The backoff is per NODE, not per half: one powered-off box must not
    freeze the deck's view of a healthy one for five minutes."""
    paced.agents["nimbus"] = _FakeAgent(raises=NodeAgentUnreachable("refused"))
    paced.half()

    paced.now += _PACED_TTL_S
    paced.half()

    assert paced.gpu_probes("nimbus") == 1
    assert paced.gpu_probes("cirrus") == 2


def test_remote_observer_recovery_resets_the_backoff(paced):
    """A node that answers again is back on the plain TTL immediately —
    its failure history described a state it is no longer in."""
    paced.agents["nimbus"] = _FakeAgent(raises=NodeAgentUnreachable("refused"))
    paced.half()
    paced.now += _PACED_BACKOFF_S
    paced.agents["nimbus"] = _FakeAgent(_pool())
    paced.half()                                   # recovered

    paced.now += _PACED_TTL_S
    paced.half()

    assert paced.gpu_probes("nimbus") == 2         # the fresh agent, twice
    assert paced.half()["remote_tenants"]["nimbus/song-r"]["state"] == "idle"


def test_remote_observer_invalidate_forces_a_fresh_probe(paced):
    """Call it after ACTING on a node (a load/unload), whose whole purpose
    is to change what this caches — SparkObserver.invalidate's rationale."""
    paced.half()

    paced.observer.invalidate()
    paced.half()

    assert paced.gpu_probes("nimbus") == 2


def test_remote_observer_default_ttl_paces_a_burst_of_ticks(hand_built_registry):
    """The DEFAULT constructor (what production uses) paces too — a test
    that only ever passed explicit TTLs would not catch a default of 0."""
    from app.node_clients import RemoteObserver

    registry = hand_built_registry(
        _paced_entries(), {"nimbus": "key-nimbus", "cirrus": "key-cirrus"})
    paced = _Paced(registry, {"nimbus": _FakeAgent(_pool()),
                              "cirrus": _FakeAgent(_pool())})
    paced.observer = RemoteObserver(clock=lambda: paced.now)

    for _ in range(5):
        paced.now += 2.0                 # five arbiter ticks
        paced.half()

    assert paced.gpu_probes("nimbus") == 1


def test_remote_observer_without_the_wiring_is_an_empty_half(paced):
    """Same posture as remote_world_half's own: a caller missing a
    dependency gets no remote engines, never a crash."""
    assert paced.observer.half(paced.registry, None, None, paced.world,
                               object()) == {"remote_gpus": {},
                                             "remote_tenants": {}}


# ===========================================================================
# INST I1 Task 7 — control-dispatching binding: "instances" nodes join
# "swap" nodes as OPERABLE, with no "local" special-casing anywhere in the
# gate. Node id "cirrus" (never "sparky" — fixture discipline).
# ===========================================================================


def test_client_for_binds_an_instances_node_without_serving_address(hand_built_registry):
    store = hand_built_registry([{"id": "cirrus", "label": "C", "agent_kind": "node-agent",
                                  "address": "http://c:7720", "control": "instances",
                                  "instance_port_range": {"start": 11500, "end": 11509}, "engines": []}],
                                {"cirrus": "k"})
    built = []
    def factory(entry, credential):
        built.append((entry["control"], credential)); return object()
    clients = NodeClients(store, factory)
    assert clients.client_for("cirrus") is not None
    assert built == [("instances", "k")]


def test_client_for_still_requires_serving_address_for_swap_only(hand_built_registry):
    store = hand_built_registry([
        {"id": "cirrus", "label": "C", "agent_kind": "node-agent",
         "address": "http://c:7720", "control": "swap", "engines": []},
        {"id": "nimbus", "label": "N", "agent_kind": "node-agent",
         "control": "instances", "engines": []},
        {"id": "orbit", "label": "O", "agent_kind": "node-agent",
         "address": "http://o:7720", "serving_address": "http://o:8000",
         "control": "none", "engines": []},
    ], {"cirrus": "k-c", "nimbus": "k-n", "orbit": "k-o"})

    def factory(entry, credential):
        return object()

    clients = NodeClients(store, factory)

    assert clients.client_for("cirrus") is None    # swap, no serving_address
    assert clients.client_for("nimbus") is None     # instances, no address
    assert clients.client_for("orbit") is None       # control:none, despite having everything


def test_observers_snapshot_keeps_instances_clients_alive(hand_built_registry):
    """NodeObservers.snapshot() builds observers for swap nodes ONLY but must
    retire clients for nodes that are neither swap NOR instances."""
    store = hand_built_registry([
        {"id": "boxa", "label": "A", "agent_kind": "node-agent",
         "address": "http://a:7720", "serving_address": "http://a:8000",
         "control": "swap", "engines": []},
        {"id": "cirrus", "label": "C", "agent_kind": "node-agent",
         "address": "http://c:7720", "control": "instances",
         "instance_port_range": {"start": 11500, "end": 11509}, "engines": []},
        {"id": "orbit", "label": "O", "agent_kind": "node-agent",
         "address": "http://o:7720", "serving_address": "http://o:8000",
         "control": "swap", "engines": []},
    ], {"boxa": "k-a", "cirrus": "k-c", "orbit": "k-o"})

    clients = NodeClients(store, FakeClient)
    observers = NodeObservers(store, clients)

    instances_client = clients.client_for("cirrus")
    # "orbit" was operable a moment ago (still is, until the patch below) --
    # build its client first so there is something for retire_absent() to
    # actually drop, then demote it out of the operable set entirely.
    orbit_client = clients.client_for("orbit")
    assert orbit_client is not None
    store.patch("orbit", control="none")

    snap = observers.snapshot()

    assert set(snap) == {"boxa"}                             # observers: swap only
    assert clients.client_for("cirrus") is instances_client  # kept alive
    assert orbit_client.closed is True                        # retired
