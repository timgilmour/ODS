"""tests/test_nodes_router.py — CRUD + test-connection + the never-echo gate.

App built via create_app() (the autouse conftest fixture gives every test an
isolated tmp data dir, so the real seeded NodeStore is safe to use). The
client factory is swapped on the deck for a fake — the test route must never
open a socket in tests.
"""
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.engines import EngineError

SECRET = "hunter2-super-secret-key"


class FakeAgent:
    def __init__(self, info=None, raises=None):
        self._info = info or {"name": "hera", "hostname": "hera", "platform": "cuda",
                              "capabilities": ["metrics"], "gpus": [{"index": 0}]}
        self._raises = raises

    def info(self):
        if self._raises:
            raise self._raises
        return self._info

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    # The tests/test_api.py idiom: no background threads under TestClient's
    # lifespan. Set per-fixture — conftest deliberately does not set it.
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    main_module._deck_by_settings_id.clear()
    app = main_module.create_app()
    app.state.deck["node_agent_client_factory"] = lambda address, key: FakeAgent()
    with TestClient(app) as c:
        yield c


def _create(client, **over):
    body = {"id": "hera", "label": "Hera Box", "address": "http://hera:7720",
            "credential": SECRET}
    body.update(over)
    return client.post("/api/nodes", json=body)


def test_create_then_list(client):
    assert _create(client).status_code == 200
    body = client.get("/api/nodes").json()
    hera = next(n for n in body["nodes"] if n["id"] == "hera")
    assert hera["credential_set"] is True
    assert hera["agent_kind"] == "node-agent"


def test_credential_never_echoed_anywhere(client):
    responses = [
        _create(client),
        client.get("/api/nodes"),
        client.get("/api/state"),
        client.post("/api/nodes/test", json={"node_id": "hera"}),
        client.put("/api/nodes/hera", json={"label": "Renamed"}),
        _create(client, id="hera"),                      # 409 error body
        client.post("/api/nodes", json={"id": "BAD ID", "label": "x",
                                        "address": "http://x:7720",
                                        "credential": SECRET}),  # 422 error body
    ]
    for resp in responses:
        assert SECRET not in resp.text, f"credential leaked in {resp.request.url}"


def test_duplicate_id_409_bad_slug_422(client):
    _create(client)
    assert _create(client).status_code == 409
    assert _create(client, id="Not A Slug").status_code == 422


def test_empty_credential_string_is_refused(client):
    # Literal-and-declared: "" is not "absent", and it is not a credential.
    assert _create(client, credential="").status_code == 422


def test_delete_local_is_refused(client):
    assert client.delete("/api/nodes/local").status_code == 409


def test_delete_removes_node(client):
    _create(client)
    assert client.delete("/api/nodes/hera").status_code == 200
    ids = [n["id"] for n in client.get("/api/nodes").json()["nodes"]]
    assert "hera" not in ids


def test_test_connection_by_stored_id(client):
    _create(client)
    body = client.post("/api/nodes/test", json={"node_id": "hera"}).json()
    assert body["ok"] is True
    assert body["gpu_count"] == 1
    assert body["platform"] == "cuda"
    assert "credential" not in body


def test_test_connection_pre_save(client):
    body = client.post("/api/nodes/test",
                       json={"address": "http://new:7720",
                             "credential": "some-key"}).json()
    assert body["ok"] is True


def test_test_connection_failure_is_ok_false_and_evented(client):
    client.app.state.deck["node_agent_client_factory"] = (
        lambda address, key: FakeAgent(raises=EngineError("401 unauthorized")))
    _create(client)
    body = client.post("/api/nodes/test", json={"node_id": "hera"}).json()
    assert body["ok"] is False and "401" in body["error"]
    kinds = [e["kind"] for e in client.get("/api/events").json()["events"]]
    assert "node-test-failed" in kinds


def test_test_connection_requires_a_target(client):
    assert client.post("/api/nodes/test", json={}).status_code == 422
