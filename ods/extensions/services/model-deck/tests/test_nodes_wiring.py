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


def test_intent_json_is_never_touched_by_node_machinery(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path / "data"))
    app = _app(monkeypatch)
    with TestClient(app) as c:
        c.get("/api/state")
        c.get("/api/nodes")
    assert not (tmp_path / "data" / "intent.json").exists()
