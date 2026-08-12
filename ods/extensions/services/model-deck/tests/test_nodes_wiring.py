"""tests/test_nodes_wiring.py — seed at build, spark from registry,
/api/state nodes block."""
import json

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.observe import spark_node_id


@pytest.fixture(autouse=True)
def _fresh_deck():
    main_module._deck_by_settings_id.clear()
    yield
    main_module._deck_by_settings_id.clear()


def _app(monkeypatch, **env):
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")  # test_api.py idiom
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return main_module.create_app()


def test_seed_runs_at_build_local_label_from_env(monkeypatch):
    app = _app(monkeypatch, MODEL_DECK_NODE_LABEL="autarch")
    with TestClient(app) as c:
        state = c.get("/api/state").json()
    assert state["node"] == {"id": "local", "label": "autarch"}
    ids = {n["id"] for n in state["nodes"]}
    assert "local" in ids


def test_spark_client_is_built_from_the_registry_seed(monkeypatch):
    app = _app(monkeypatch,
               MODEL_DECK_SPARK_NODE_URL="http://192.168.1.7:7720",
               MODEL_DECK_SPARK_SERVING_URL="http://192.168.1.7:8000",
               ODS_REMOTE_NODE_KEYS=json.dumps({"sparky": "spark-key"}))
    deck = app.state.deck
    assert deck["spark"] is not None
    entry = deck["node_store"].get(spark_node_id())
    assert entry["address"] == "http://192.168.1.7:7720"
    # The client was built FROM the registry: mutate env after build has no
    # path back in (seed-once), which the next test pins.


def test_registry_wins_over_changed_env_after_seed(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path / "data"))
    app1 = _app(monkeypatch,
                MODEL_DECK_SPARK_NODE_URL="http://192.168.1.7:7720",
                MODEL_DECK_SPARK_SERVING_URL="http://192.168.1.7:8000",
                ODS_REMOTE_NODE_KEYS=json.dumps({"sparky": "spark-key"}))
    assert app1.state.deck["spark"] is not None
    main_module._deck_by_settings_id.clear()
    # Same data dir, env now points elsewhere: the registry entry wins.
    app2 = _app(monkeypatch, MODEL_DECK_SPARK_NODE_URL="http://elsewhere:7720")
    entry = app2.state.deck["node_store"].get(spark_node_id())
    assert entry["address"] == "http://192.168.1.7:7720"


def test_no_spark_env_no_spark_client_registry_still_seeds_local(monkeypatch):
    app = _app(monkeypatch)
    assert app.state.deck["spark"] is None
    assert app.state.deck["node_store"].get("local") is not None


def test_nodes_block_shape_without_observer(monkeypatch):
    # MODEL_DECK_NO_WATCHER=1 decks (every unit test) have no observer:
    # status must be null for node-agent nodes, "online" for local.
    app = _app(monkeypatch,
               MODEL_DECK_SPARK_NODE_URL="http://192.168.1.7:7720",
               MODEL_DECK_SPARK_SERVING_URL="http://192.168.1.7:8000",
               ODS_REMOTE_NODE_KEYS=json.dumps({"sparky": "spark-key"}))
    with TestClient(app) as c:
        nodes = {n["id"]: n for n in c.get("/api/state").json()["nodes"]}
    assert nodes["local"]["status"] == "online"
    assert nodes["sparky"]["status"] is None
    assert nodes["sparky"]["credential_set"] is True
    assert "credential" not in nodes["sparky"]
    assert nodes["sparky"]["address"] == "http://192.168.1.7:7720"


def test_state_200s_with_a_malformed_element_in_nodes_json(monkeypatch):
    # _nodes_block (app/routers/status.py) indexes entry["label"]/
    # entry["agent_kind"] directly, trusting NodeStore's element-level
    # boundary gate (app/node_store.py._load()). A hand-edited malformed
    # element must be dropped there, not 500 this endpoint.
    app = _app(monkeypatch, MODEL_DECK_NODE_LABEL="autarch")
    node_store = app.state.deck["node_store"]
    data = json.loads(node_store._path.read_text())
    data.append({"id": "ghost", "label": 999, "agent_kind": "node-agent",
                "address": "http://ghost:7720"})   # non-string label: dropped
    node_store._path.write_text(json.dumps(data))
    with TestClient(app) as c:
        resp = c.get("/api/state")
    assert resp.status_code == 200
    ids = {n["id"] for n in resp.json()["nodes"]}
    assert "local" in ids
    assert "ghost" not in ids


def test_intent_json_is_never_touched_by_node_machinery(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path / "data"))
    app = _app(monkeypatch)
    with TestClient(app) as c:
        c.get("/api/state")
        c.get("/api/nodes")
    assert not (tmp_path / "data" / "intent.json").exists()


# --- actuation binding staleness [max-review #13] ---------------------------
#
# main.py builds the SparkClient ONCE, at app build, from the registry (see
# its comment there). Connection edits reach OBSERVATION immediately —
# NodeObserver re-reads the registry every tick — but ACTUATION keeps using
# the boot-time address until a restart. That consequence was documented in
# main.py and surfaced to the operator NOWHERE. These pin the surfacing.

_SPARK_ENV = {
    "MODEL_DECK_SPARK_NODE_URL": "http://192.168.1.7:7720",
    "MODEL_DECK_SPARK_SERVING_URL": "http://192.168.1.7:8000",
    "ODS_REMOTE_NODE_KEYS": json.dumps({"sparky": "spark-key"}),
}


def _spark_node(client):
    nodes = client.get("/api/nodes").json()["nodes"]
    return next(n for n in nodes if n["id"] == spark_node_id())


def test_nodes_list_not_stale_when_registry_matches_binding(monkeypatch):
    app = _app(monkeypatch, **_SPARK_ENV)
    assert app.state.deck["spark"] is not None  # fixture precondition: bound
    with TestClient(app) as c:
        assert _spark_node(c)["actuation_stale"] is False


def test_nodes_list_flags_stale_actuation_after_address_edit(monkeypatch):
    """The finding: the bound client still points at the boot address."""
    app = _app(monkeypatch, **_SPARK_ENV)
    with TestClient(app) as c:
        c.put(f"/api/nodes/{spark_node_id()}",
              json={"address": "http://192.168.1.99:7720"})
        assert _spark_node(c)["actuation_stale"] is True


def test_nodes_list_flags_stale_actuation_after_serving_address_edit(monkeypatch):
    """serving_address is bound too (main.py passes it as serving_url), so it
    goes stale by the same mechanism — asserted separately because a
    binding record that only captured `address` would pass the test above."""
    app = _app(monkeypatch, **_SPARK_ENV)
    with TestClient(app) as c:
        c.put(f"/api/nodes/{spark_node_id()}",
              json={"serving_address": "http://192.168.1.7:9999"})
        assert _spark_node(c)["actuation_stale"] is True


def test_nodes_list_flags_stale_actuation_after_credential_edit(monkeypatch):
    """The credential is bound as node_key. Compared by fingerprint — the
    value never leaves the store — and the response still must not echo it."""
    app = _app(monkeypatch, **_SPARK_ENV)
    with TestClient(app) as c:
        c.put(f"/api/nodes/{spark_node_id()}", json={"credential": "rotated-key"})
        node = _spark_node(c)
        assert node["actuation_stale"] is True
        assert "rotated-key" not in c.get("/api/nodes").text


def test_nodes_list_not_stale_when_an_edit_changes_nothing(monkeypatch):
    """Re-submitting the SAME address is not a rebind-worthy change. Pins that
    the flag compares CONFIGURATION, not "was a PUT ever issued"."""
    app = _app(monkeypatch, **_SPARK_ENV)
    with TestClient(app) as c:
        c.put(f"/api/nodes/{spark_node_id()}",
              json={"address": "http://192.168.1.7:7720", "label": "renamed"})
        assert _spark_node(c)["actuation_stale"] is False


def test_nodes_list_flags_stale_when_spark_configured_after_boot(monkeypatch):
    """Booted with no spark env => no client bound at all. Configuring one
    now cannot actuate until a restart, so it is stale rather than fine."""
    app = _app(monkeypatch)  # no spark env
    assert app.state.deck["spark"] is None
    with TestClient(app) as c:
        c.post("/api/nodes", json={"id": spark_node_id(), "label": "sparky",
                                   "address": "http://192.168.1.7:7720",
                                   "serving_address": "http://192.168.1.7:8000",
                                   "credential": "spark-key"})
        assert _spark_node(c)["actuation_stale"] is True


def test_unbound_and_unconfigured_spark_is_not_stale(monkeypatch):
    """No client, no configuration: nothing is out of date, so no nag."""
    app = _app(monkeypatch)
    with TestClient(app) as c:
        c.post("/api/nodes", json={"id": spark_node_id(), "label": "sparky",
                                   "address": "http://192.168.1.7:7720"})
        # serving_address and credential absent -> a restart would bind nothing
        assert _spark_node(c)["actuation_stale"] is False


def test_non_spark_nodes_are_never_stale(monkeypatch):
    """Observe-only nodes have no actuation binding to go stale."""
    app = _app(monkeypatch, **_SPARK_ENV)
    with TestClient(app) as c:
        c.post("/api/nodes", json={"id": "hera", "label": "Hera",
                                   "address": "http://hera:7720",
                                   "credential": "k"})
        c.put("/api/nodes/hera", json={"address": "http://hera:9999"})
        nodes = {n["id"]: n for n in c.get("/api/nodes").json()["nodes"]}
        assert nodes["hera"]["actuation_stale"] is False
        assert nodes["local"]["actuation_stale"] is False


def test_state_nodes_block_carries_actuation_stale(monkeypatch):
    """/api/state is what the Nodes SCREEN renders (ui/src/components/
    NodesView.tsx takes DeckNodeEntry[], not the registry CRUD list), so the
    flag has to reach this block or the finding stays exactly as invisible as
    it was. Pinned separately from the /api/nodes tests above because those
    would all still pass with this block unchanged."""
    app = _app(monkeypatch, **_SPARK_ENV)
    with TestClient(app) as c:
        before = {n["id"]: n for n in c.get("/api/state").json()["nodes"]}
        assert before[spark_node_id()]["actuation_stale"] is False
        assert before["local"]["actuation_stale"] is False

        c.put(f"/api/nodes/{spark_node_id()}",
              json={"address": "http://192.168.1.99:7720"})

        after = {n["id"]: n for n in c.get("/api/state").json()["nodes"]}
        assert after[spark_node_id()]["actuation_stale"] is True
        assert after["local"]["actuation_stale"] is False


def test_deck_carries_node_clients_and_observers(tmp_path, monkeypatch):
    from tests.test_api import make_app
    app, deck = make_app(tmp_path, monkeypatch)
    from app.node_clients import NodeClients, NodeObservers
    assert isinstance(deck["node_clients"], NodeClients)
    assert isinstance(deck["node_observers"], NodeObservers)


def test_boot_stamps_control_on_legacy_seeded_spark(tmp_path, monkeypatch):
    """A pre-N1 install — nodes.json already on disk (written before the
    `control` field existed) with a sparky entry carrying the three swap
    prerequisites — comes up with control:"swap" stamped on disk, exactly
    once (design §8).

    Deliberately pre-populates nodes.json/node_credentials.json directly
    (the `_legacy_file` idiom tests/test_node_store.py uses) rather than
    relying on MODEL_DECK_SPARK_NODE_URL/etc + seed_if_missing: on a
    genuinely fresh box (no nodes.json yet) seed_if_missing's own add()
    call stamps control:"none" on the entry it creates (NodeStore.add's
    documented default), so stamp_missing_control's "control" key is
    already present and it never promotes anything — env-seeding a brand
    new box is not the same event as upgrading a box that already has
    data on disk from before the control field shipped, which is what
    this test — and stamp_missing_control's own docstring ("pre-N1
    files") — is about."""
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    (tmp_path / "nodes.json").write_text(json.dumps([
        {"id": "local", "label": "local", "agent_kind": "local"},
        {"id": "sparky", "label": "sparky", "agent_kind": "node-agent",
         "address": "http://sparky:7720", "serving_address": "http://sparky:8000"},
    ]))
    (tmp_path / "node_credentials.json").write_text(json.dumps({"sparky": "k"}))
    from app.main import create_app
    app = create_app()
    entry = app.state.deck["node_store"].get("sparky")
    assert entry["control"] == "swap"


# --- harvest routes per swap node (N1 T11) ----------------------------------
#
# main._build_deck builds one (node_id, "vllm") route per control:"swap"
# registry entry (design §4's disclosed limitation: the PAIR SET is
# enumerated at app build, the CLIENT behind each pair is live-rebound
# through node_clients). Prepares nodes.json/node_credentials.json via a
# NodeStore pointed at the same data dir BEFORE create_app() — the loop that
# builds routes only runs at build, so the store has to be seeded first,
# mirroring test_boot_stamps_control_on_legacy_seeded_spark above.


def test_routes_built_per_swap_node(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    from app.node_store import NodeStore

    store = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")
    store.add({"id": "local", "label": "local", "agent_kind": "local"})
    for node_id in ("boxa", "boxb"):
        store.add({"id": node_id, "label": node_id, "agent_kind": "node-agent",
                   "address": f"http://{node_id}:7720",
                   "serving_address": f"http://{node_id}:8000",
                   "control": "swap"},
                  credential="k")

    from app.main import create_app
    app = create_app()
    deck = app.state.deck

    assert deck["configurable_engines"] == [("boxa", "vllm"), ("boxb", "vllm")]
    assert deck["engine_exec"] is not None


def test_none_control_swap_capable_node_yields_no_route(tmp_path, monkeypatch):
    """A node with every swap prerequisite present (address, serving_address,
    credential) but control left at "none" gets no harvest pair -- control is
    the gate, not mere capability."""
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    from app.node_store import NodeStore

    store = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")
    store.add({"id": "local", "label": "local", "agent_kind": "local"})
    store.add({"id": "boxc", "label": "boxc", "agent_kind": "node-agent",
              "address": "http://boxc:7720", "serving_address": "http://boxc:8000",
              "control": "none"},
              credential="k")

    from app.main import create_app
    app = create_app()
    deck = app.state.deck

    assert deck["configurable_engines"] == []
    assert deck["engine_exec"] is None


def test_addresses_without_a_credential_is_not_stale(monkeypatch):
    """CLOSES A MUTATION-DEAD TERM [T8 review]. seed_if_missing seeds spark
    with BOTH addresses and NO credential when ODS_REMOTE_NODE_KEYS is absent
    or malformed (node_store.py's documented degradation), so this state is
    real, not hypothetical. A restart would bind no client either — nothing
    is out of date, so nothing is stale.

    The sibling unbound test can't cover this: its fixture omits
    serving_address AND the credential, so the credential term is never the
    deciding one. Two mutants (dropping `and current["credential_fp"]`, and
    making credential_fingerprint digest "" rather than return None) each
    passed the whole suite before this test existed — and each would give the
    operator a PERMANENT false "Restart required" banner.
    """
    app = _app(monkeypatch,
               MODEL_DECK_SPARK_NODE_URL="http://192.168.1.7:7720",
               MODEL_DECK_SPARK_SERVING_URL="http://192.168.1.7:8000")
    assert app.state.deck["spark"] is None       # no key -> no client bound
    with TestClient(app) as c:
        node = _spark_node(c)
        assert node["credential_set"] is False
        assert node["actuation_stale"] is False
