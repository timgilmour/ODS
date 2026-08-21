import json, pytest
from fastapi.testclient import TestClient
from app.engines import BusyError, EngineError
from tests.test_api import make_app

class _Recorder:
    def __init__(self, fail=None): self.calls, self.fail = [], fail
    def request(self, verb, document):
        self.calls.append((verb, document))
        if self.fail: raise self.fail
    def close(self): pass

def _instances_node(deck, client, observed=(0, 1)):
    store = deck["node_store"]
    store.update("local", {"address": "http://172.18.0.1:7720",
                           "instance_port_range": {"start": 11500, "end": 11501}}, credential="k")
    store.update("local", {"control": "instances"})
    class _Clients:
        def client_for(self, node_id): return client if node_id == "local" else None
    deck["node_clients"] = _Clients()

BODY = {"kind": "hipfire", "gpu_indices": [1], "env": {"HIPFIRE_MODEL": "qwen3.8:27b"}}

def test_create_writes_a_managed_entry_then_ships_the_document(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch); rec = _Recorder(); _instances_node(deck, rec)
    r = TestClient(app).post("/api/nodes/local/instances", json=BODY)
    assert r.status_code == 201, r.text
    e = r.json()
    assert e["resource"] == "hipfire-1" and e["managed"] is True and e["port"] == 11500
    assert e["connection"] == {"container": "deck-hipfire-1"}
    assert e["container_consent"] is True and e["gpu_indices"] == [1]
    assert e["policy_defaults"] == {"priority": 100, "pinned": False, "idle_ttl": 0}
    assert rec.calls == [("create", {"resource": "hipfire-1", "kind": "hipfire", "gpu_indices": [1],
                                     "port": 11500, "env": {"HIPFIRE_MODEL": "qwen3.8:27b"}})]
    assert any(e["resource"] == "hipfire-1" for e in deck["node_store"].get("local")["engines"])
    ev = [json.loads(l) for l in (deck["events_path"]).read_text().splitlines()]
    assert ev[-1]["kind"] == "instance-created" and ev[-1]["detail"]["port"] == 11500

def test_create_rolls_back_the_declaration_when_the_agent_refuses(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch); _instances_node(deck, _Recorder(fail=EngineError("boom")))
    r = TestClient(app).post("/api/nodes/local/instances", json=BODY)
    assert r.status_code == 502 and "boom" in r.text
    assert not any(e["resource"] == "hipfire-1" for e in deck["node_store"].get("local")["engines"])
    ev = [json.loads(l) for l in (deck["events_path"]).read_text().splitlines()]
    assert ev[-1]["kind"] == "instance-create-failed"
    # Interface block (task brief): instance-create-failed carries
    # {node,resource,kind,error} — the ONE failure event with `kind`
    # (remove/move failures don't name it, matching their own spec rows).
    assert ev[-1]["detail"] == {"node": "local", "resource": "hipfire-1", "kind": "hipfire", "error": "boom"}

def test_create_refusals(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch); _instances_node(deck, _Recorder()); c = TestClient(app)
    assert c.post("/api/nodes/local/instances", json={**BODY, "kind": "sglang-omni"}).status_code == 422   # not instantiable
    assert c.post("/api/nodes/local/instances", json={**BODY, "gpu_indices": [0, 1]}).status_code == 422   # max_gpus 1
    assert c.post("/api/nodes/local/instances", json={**BODY, "gpu_indices": [7]}).status_code == 422      # unobserved
    assert c.post("/api/nodes/local/instances", json={**BODY, "env": {}}).status_code == 422              # HIPFIRE_MODEL required
    c.post("/api/nodes/local/instances", json=BODY); c.post("/api/nodes/local/instances", json=BODY)
    r = c.post("/api/nodes/local/instances", json=BODY)                                                   # range 11500-11501 exhausted
    assert r.status_code == 409 and "no free port" in r.text

def test_create_on_a_non_instances_node_is_503_naming_the_prereqs(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    r = TestClient(app).post("/api/nodes/local/instances", json=BODY)
    # r.json()["detail"], not r.text: the raw wire body is JSON, so a
    # literal `"` in the message is backslash-escaped on the wire
    # (`\"instances\"`) — checking r.text for an UNescaped `"control:
    # "instances""` substring can never pass for any HTTPException detail
    # that itself contains a quoted word (json.dumps escapes every `"`
    # inside a string value, no way around it). Decoding first checks the
    # same content the brief's assertion intended.
    assert r.status_code == 503 and 'control: "instances"' in r.json()["detail"]

def test_create_missing_port_range_is_503_not_a_keyerror_500(tmp_path, monkeypatch):
    # node_store._require_instances_prereqs is the write-side gate and
    # refuses a PATCH that sets control:"instances" without a range — so a
    # node missing it can only exist via a raw nodes.json (hand-edited, or
    # written before this field existed). NodeStore._load() re-reads the
    # file on every call (no cache to invalidate), so writing straight to
    # disk is enough; no explicit reload step needed.
    app, deck = make_app(tmp_path, monkeypatch)
    _instances_node(deck, _Recorder())
    store = deck["node_store"]
    data = json.loads(store._path.read_text())
    for entry in data:
        if entry["id"] == "local":
            del entry["instance_port_range"]
    store._path.write_text(json.dumps(data))
    r = TestClient(app).post("/api/nodes/local/instances", json=BODY)
    assert r.status_code == 503 and "instance_port_range" in r.json()["detail"]

def test_remove_holds_ships_then_forgets_declaration_intent_policy(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch); rec = _Recorder(); _instances_node(deck, rec); c = TestClient(app)
    c.post("/api/nodes/local/instances", json=BODY)
    deck["intent_store"].record("local/hipfire-1", state="loaded", model=None, engine="hipfire")
    r = c.delete("/api/nodes/local/instances/hipfire-1")
    assert r.status_code == 200
    assert rec.calls[-1][0] == "remove"
    assert deck["hold_store"].held("local/hipfire-1")
    assert "local/hipfire-1" not in deck["intent_store"].get()
    assert not any(e["resource"] == "hipfire-1" for e in deck["node_store"].get("local")["engines"])
    assert "hipfire-1" not in deck["policy_store"].get()

def test_remove_failure_releases_the_hold(tmp_path, monkeypatch):
    # A hold announces "this absence is ours, don't restore it" — but if
    # the teardown never shipped, nothing became absent. Leaving the hold
    # in place would silence the reconciler on a container that is still
    # there for up to _HOLD_S with no actuator left watching it.
    app, deck = make_app(tmp_path, monkeypatch); rec = _Recorder(); _instances_node(deck, rec); c = TestClient(app)
    c.post("/api/nodes/local/instances", json=BODY)
    rec.fail = EngineError("boom")
    r = c.delete("/api/nodes/local/instances/hipfire-1")
    assert r.status_code == 502
    assert not deck["hold_store"].held("local/hipfire-1")

def test_remove_refuses_an_unmanaged_declaration(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch); _instances_node(deck, _Recorder())
    r = TestClient(app).delete("/api/nodes/local/instances/hipfire")     # the seeded, managed:false entry
    assert r.status_code == 409 and "not a deck-managed instance" in r.text

def test_move_ships_new_claim_then_updates_declaration_and_forgets_intent(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch); rec = _Recorder(); _instances_node(deck, rec); c = TestClient(app)
    c.post("/api/nodes/local/instances", json=BODY)
    deck["intent_store"].record("local/hipfire-1", state="loaded", model=None, engine="hipfire")
    r = c.post("/api/nodes/local/instances/hipfire-1/move", json={"gpu_indices": [0]})
    assert r.status_code == 200 and r.json()["gpu_indices"] == [0]
    assert rec.calls[-1] == ("move", {"resource": "hipfire-1", "kind": "hipfire", "gpu_indices": [0], "port": 11500,
                                      "env": {"HIPFIRE_MODEL": "qwen3.8:27b"}})
    assert "local/hipfire-1" not in deck["intent_store"].get()
    assert c.post("/api/nodes/local/instances/hipfire-1/move", json={"gpu_indices": [0]}).status_code == 409  # unchanged

def test_busy_agent_is_409_not_502(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch); _instances_node(deck, _Recorder(fail=BusyError("pending")))
    assert TestClient(app).post("/api/nodes/local/instances", json=BODY).status_code == 409
