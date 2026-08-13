"""NodeClients/NodeObservers — the lazy self-healing rebind (design §3).

Fixtures use boxa/boxb, labels ≠ ids, plus one control:"none" node WITH a
serving_address ([[defaults-that-hide-bugs]]): operability must come from
the declaration alone, never inferred from data presence.
"""

import httpx
import pytest

from app.node_clients import NodeClients, NodeObservers
from app.node_store import NodeStore


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
