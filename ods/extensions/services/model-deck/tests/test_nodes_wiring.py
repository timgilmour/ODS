"""tests/test_nodes_wiring.py — seed at build, spark from registry,
/api/state nodes block."""
import json

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.node_store import LEGACY_SPARK_SEED_ID


@pytest.fixture(autouse=True)
def _fresh_deck():
    main_module._deck_by_settings_id.clear()
    yield
    main_module._deck_by_settings_id.clear()


class _FakeTelemetry:
    """Stand-in for app.telemetry.LocalTelemetry — the FakeTelemetry idiom
    tests/test_api.py:118 states outright ("no real create_app() build should
    ever let a test hit dashboard-api over the network"). `_app` installs one
    with no rows by default: without it, every test here that calls
    /api/state fires a live GET at http://dashboard-api:3002 and passes only
    because the connection error degrades to None."""

    def __init__(self, rows=None):
        self._rows = rows

    def gpus(self):
        return self._rows


def _app(monkeypatch, **env):
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")  # test_api.py idiom
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = main_module.create_app()
    # Default stub; a test that asserts on telemetry replaces it.
    app.state.deck["telemetry"] = _FakeTelemetry(None)
    return app


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
    assert deck["node_clients"].client_for(LEGACY_SPARK_SEED_ID) is not None
    entry = deck["node_store"].get(LEGACY_SPARK_SEED_ID)
    assert entry["address"] == "http://192.168.1.7:7720"
    # The client is built FROM the registry: mutate env after build has no
    # path back in (seed-once), which the next test pins.


def test_registry_wins_over_changed_env_after_seed(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path / "data"))
    app1 = _app(monkeypatch,
                MODEL_DECK_SPARK_NODE_URL="http://192.168.1.7:7720",
                MODEL_DECK_SPARK_SERVING_URL="http://192.168.1.7:8000",
                ODS_REMOTE_NODE_KEYS=json.dumps({"sparky": "spark-key"}))
    assert app1.state.deck["node_clients"].client_for(LEGACY_SPARK_SEED_ID) is not None
    main_module._deck_by_settings_id.clear()
    # Same data dir, env now points elsewhere: the registry entry wins.
    app2 = _app(monkeypatch, MODEL_DECK_SPARK_NODE_URL="http://elsewhere:7720")
    entry = app2.state.deck["node_store"].get(LEGACY_SPARK_SEED_ID)
    assert entry["address"] == "http://192.168.1.7:7720"


def test_no_spark_env_no_spark_client_registry_still_seeds_local(monkeypatch):
    app = _app(monkeypatch)
    assert app.state.deck["node_clients"].client_for(LEGACY_SPARK_SEED_ID) is None
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
    # N1 T12: status.py's _nodes_block now declares control explicitly
    # (the heal in NodeStore._load guarantees the key's presence on every
    # entry) — the env seed stamps "swap" here because a credential is set.
    assert nodes["local"]["control"] == "none"
    assert nodes["sparky"]["control"] == "swap"


class _FakeNodeObserver:
    """Stand-in for app.node_observer.NodeObserver: a fixed snapshot, no
    thread. Used here only to prove telemetry doesn't touch the remote
    (node-agent) arm of _nodes_block, which stays sourced from THIS."""

    def __init__(self, snap):
        self._snap = snap

    def snapshot(self):
        return self._snap


def test_local_gpus_come_from_telemetry_remote_gpus_are_untouched(monkeypatch):
    app = _app(monkeypatch,
               MODEL_DECK_SPARK_NODE_URL="http://192.168.1.7:7720",
               MODEL_DECK_SPARK_SERVING_URL="http://192.168.1.7:8000",
               ODS_REMOTE_NODE_KEYS=json.dumps({"sparky": "spark-key"}))
    deck = app.state.deck
    local_rows = [{"index": 0, "name": "Local GPU"}]
    remote_rows = [{"index": 0, "name": "Remote GPU"}]
    deck["telemetry"] = _FakeTelemetry(local_rows)
    deck["node_observer"] = _FakeNodeObserver({
        "sparky": {"status": "online", "last_seen": "t", "gpus": remote_rows,
                   "serving": None, "error": None},
    })

    with TestClient(app) as c:
        nodes = {n["id"]: n for n in c.get("/api/state").json()["nodes"]}

    # Local: from app.telemetry, NOT the (empty, for "local") observer snap.
    assert nodes["local"]["gpus"] == local_rows
    # Remote: still from the node observer, untouched by telemetry wiring.
    assert nodes["sparky"]["gpus"] == remote_rows


def test_local_gpus_is_none_when_telemetry_fetch_fails(monkeypatch):
    app = _app(monkeypatch, MODEL_DECK_NODE_LABEL="autarch")
    app.state.deck["telemetry"] = _FakeTelemetry(None)

    with TestClient(app) as c:
        nodes = {n["id"]: n for n in c.get("/api/state").json()["nodes"]}

    assert nodes["local"]["gpus"] is None


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


def test_boot_stamps_comfyui_container_on_a_pre_existing_entry(tmp_path, monkeypatch):
    """Finding 1c (whole-branch review): an already-E1 box's nodes.json —
    written before comfyui's schema accepted connection.container (Finding
    1a) — gains the settings default on disk at boot, exactly once, mirroring
    test_boot_stamps_control_on_legacy_seeded_spark above."""
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_DECK_COMFYUI_CONTAINER", "test-ods-comfyui")
    (tmp_path / "nodes.json").write_text(json.dumps([
        {"id": "local", "label": "local", "agent_kind": "local",
         "engines": [
             {"resource": "comfyui", "kind": "comfyui",
              "connection": {"url": "http://comfyui:8188"}, "gpu_index": 1,
              "container_consent": True,
              "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
         ]},
    ]))
    from app.main import create_app
    app = create_app()
    engines = {e["resource"]: e for e in app.state.deck["node_store"].get("local")["engines"]}
    assert engines["comfyui"]["connection"]["container"] == "test-ods-comfyui"


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


def test_addresses_without_a_credential_seed_control_none_and_no_client(monkeypatch):
    """seed_if_missing seeds spark with BOTH addresses and NO credential when
    ODS_REMOTE_NODE_KEYS is absent or malformed (node_store.py's documented
    degradation), so this state is real, not hypothetical: control stays
    "none" (§8's migration only promotes to "swap" when a credential is
    present) and node_clients.client_for answers None rather than building a
    client missing its third prerequisite."""
    app = _app(monkeypatch,
               MODEL_DECK_SPARK_NODE_URL="http://192.168.1.7:7720",
               MODEL_DECK_SPARK_SERVING_URL="http://192.168.1.7:8000")
    deck = app.state.deck
    assert deck["node_clients"].client_for(LEGACY_SPARK_SEED_ID) is None
    with TestClient(app) as c:
        node = next(n for n in c.get("/api/nodes").json()["nodes"]
                   if n["id"] == LEGACY_SPARK_SEED_ID)
        assert node["credential_set"] is False
        assert node["control"] == "none"
