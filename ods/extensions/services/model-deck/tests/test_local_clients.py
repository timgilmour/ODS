"""LocalClients — the lazy self-healing rebind (design §3 / Task 3),
app.node_clients.NodeClients' counterpart for the local node's declared
`engines[]` instead of swap nodes' registry rows.

Mirrors tests/test_node_clients.py's approach: a real NodeStore (not a
fake — the declaration IS what LocalClients reads live), a real
LocalClients, and construction-time proof via `isinstance` (a real
LemonadeClient, not a recording double) rather than a client_factory
injection seam, since — unlike NodeClients — LocalClients has no injected
factory: construction dispatches to `app.engine_kinds.ENGINE_KINDS[kind]
.build_client(...)` (review fix, T3 round 2), so there is nothing here to
substitute a fake for; the interesting behavior under test is entirely the
cache/rebuild/retire lifecycle around that dispatch, not the dispatch
target itself (app/local_clients.py's own module-level smoke checks and
tests/test_engine_kinds.py's adapter-surface tests already cover
build_client's per-kind construction).

Fixture rule: resource "gguf-a" (kind must be a REAL registered kind —
construction needs one to dispatch through — "lemonade" is the only
choice that isn't itself a live-topology-matching name), GPU index 2 (not
0/1).
"""

from app.engines.lemonade import LemonadeClient
from app.local_clients import LocalClients
from app.node_store import NodeStore
from app.settings import Settings


def _engine(**over):
    e = {"resource": "gguf-a", "kind": "lemonade",
         "connection": {"url": "http://gguf-a:8080",
                        "metrics_url": "http://gguf-a:8001/metrics",
                        "container": "ods-gguf-a"},
         "gpu_index": 2,
         "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60}}
    e.update(over)
    return e


def _store(tmp_path, engines=None):
    s = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")
    s.add({"id": "local", "label": "test-box", "agent_kind": "local"})
    if engines is not None:
        s.update("local", {"engines": engines})
    return s


def _settings():
    # Away from production defaults ([[defaults-that-hide-bugs]]) so a
    # test asserting on a built client's derived fields can't pass by
    # coincidence.
    return Settings(lemonade_key="test-lemonade-key")


def test_client_for_declared_resource(tmp_path):
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    client = clients.client_for("gguf-a")

    assert isinstance(client, LemonadeClient)


def test_undeclared_resource_gets_none(tmp_path):
    store = _store(tmp_path, engines=[])
    clients = LocalClients(store, _settings())

    assert clients.client_for("gguf-a") is None


def test_unknown_resource_gets_none(tmp_path):
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    assert clients.client_for("ghost") is None


def test_unchanged_declaration_returns_same_client_object(tmp_path):
    """(a) cache hit: an unchanged declaration must not rebuild — a rebuild
    on every call would defeat the point of a lazy CACHE (and, for a real
    engine client, would mean opening a fresh httpx.Client every tick)."""
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    first = clients.client_for("gguf-a")
    second = clients.client_for("gguf-a")

    assert first is second


def test_connection_change_rebuilds_the_client(tmp_path):
    """(b) a declared connection edit must rebuild — the OLD client would
    otherwise keep talking to a stale address forever."""
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    first = clients.client_for("gguf-a")
    store.update("local", {"engines": [
        _engine(connection={"url": "http://gguf-a-moved:8080",
                            "metrics_url": "http://gguf-a-moved:8001/metrics",
                            "container": "ods-gguf-a"})
    ]})
    second = clients.client_for("gguf-a")

    assert second is not first
    assert isinstance(second, LemonadeClient)


def test_policy_defaults_only_change_does_not_rebuild(tmp_path):
    """A declaration edit that leaves (kind, connection) untouched — only
    policy_defaults/gpu_index changed — must NOT rebuild: the build key is
    deliberately narrower than the whole entry (mirrors
    test_node_clients.py's test_label_change_does_not_rebuild)."""
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    first = clients.client_for("gguf-a")
    store.update("local", {"engines": [_engine(gpu_index=3)]})
    second = clients.client_for("gguf-a")

    assert second is first


def test_resource_removed_from_declaration_returns_none(tmp_path):
    """A resource dropped from the declaration is not operable — the same
    "not declared" answer as one that was never declared at all."""
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    assert clients.client_for("gguf-a") is not None
    store.update("local", {"engines": []})

    assert clients.client_for("gguf-a") is None


def test_retire_absent_drops_a_built_client(tmp_path):
    """(c) retire_absent drops the cached entry even though the DECLARATION
    itself never changes here — the removal/rename path a caller (e.g.
    app.arbiter.Watcher.tick, following the NodeObservers precedent) drives
    explicitly rather than waiting for the next client_for miss to notice.
    The declaration still matches the OLD build key throughout: if
    retire_absent were a no-op, client_for would keep returning the SAME
    cached object (test_unchanged_declaration_returns_same_client_object's
    exact scenario) — getting a NEW one back proves retire_absent, not a
    declaration edit, forced the rebuild."""
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    first = clients.client_for("gguf-a")
    clients.retire_absent(set())  # "gguf-a" is not in keep_resources
    second = clients.client_for("gguf-a")

    assert second is not first
    assert isinstance(second, LemonadeClient)


def test_retire_absent_keeps_a_client_still_in_keep_resources(tmp_path):
    store = _store(tmp_path, engines=[_engine()])
    clients = LocalClients(store, _settings())

    first = clients.client_for("gguf-a")
    clients.retire_absent({"gguf-a"})
    second = clients.client_for("gguf-a")

    assert second is first
