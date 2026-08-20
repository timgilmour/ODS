import json
from fastapi.testclient import TestClient
import pytest

import instances
import nodeconfig
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}
DOC = {"resource": "gguf-a", "kind": "lemonade", "gpu_indices": [3, 4], "port": 11500,
       "env": {"LEMONADE_CTX_SIZE": "32768"}}


def _enable(monkeypatch, tmp_path):
    monkeypatch.setattr(nodeconfig, "NODE_INSTANCES_CTL_DIR", str(tmp_path))
    return tmp_path


def test_validate_document_exact_keys_and_shapes():
    assert instances.validate_document(DOC) == DOC
    for bad, why in (({**DOC, "extra": 1}, "exactly the keys"),
                     ({**DOC, "resource": "Bad_Name"}, "resource must match"),
                     ({**DOC, "gpu_indices": []}, "non-empty"),
                     ({**DOC, "gpu_indices": [True]}, "non-negative integers"),
                     ({**DOC, "port": 80}, "1024-65535"),
                     ({**DOC, "env": {"A": 1}}, "env values must be strings"),
                     ({**DOC, "kind": ""}, "kind must be a non-empty string")):
        with pytest.raises(ValueError, match=why):
            instances.validate_document(bad)


def test_disabled_answers_503_and_info_omits_capability(monkeypatch):
    monkeypatch.setattr(nodeconfig, "NODE_INSTANCES_CTL_DIR", "")
    r = client.post("/v1/node/instance/gguf-a", json={"verb": "create", "document": DOC}, headers=AUTH)
    assert r.status_code == 503
    assert "instances" not in client.get("/v1/node/info", headers=AUTH).json()["capabilities"]


def test_accepted_request_is_written_atomically_and_capacity_is_one(monkeypatch, tmp_path):
    ctl = _enable(monkeypatch, tmp_path)
    r = client.post("/v1/node/instance/gguf-a", json={"verb": "create", "document": DOC}, headers=AUTH)
    assert r.status_code == 202 and r.json() == {"accepted": True}
    req = json.loads((ctl / "instance-req.json").read_text())
    assert req["verb"] == "create" and req["document"] == DOC and isinstance(req["ts"], float)
    assert not list(ctl.glob(".instance-req.*.tmp"))
    r2 = client.post("/v1/node/instance/gguf-a", json={"verb": "remove", "document": DOC}, headers=AUTH)
    assert r2.status_code == 409
    assert "instances" in client.get("/v1/node/info", headers=AUTH).json()["capabilities"]


def test_path_resource_must_match_the_document(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    r = client.post("/v1/node/instance/other", json={"verb": "create", "document": DOC}, headers=AUTH)
    assert r.status_code == 422 and "path resource 'other' != document resource 'gguf-a'" in r.text


def test_unknown_verb_is_422(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    r = client.post("/v1/node/instance/gguf-a", json={"verb": "restart", "document": DOC}, headers=AUTH)
    assert r.status_code == 422 and "verb must be one of" in r.text


def test_status_reads_the_helpers_result_or_null(monkeypatch, tmp_path):
    ctl = _enable(monkeypatch, tmp_path)
    assert client.get("/v1/node/instance/gguf-a/status", headers=AUTH).json() == {"result": None}
    (ctl / "instance-status-gguf-a.json").write_text(json.dumps({"resource": "gguf-a", "verb": "create", "ok": True, "error": None, "ts": "t"}))
    assert client.get("/v1/node/instance/gguf-a/status", headers=AUTH).json()["result"]["ok"] is True
    (ctl / "instance-status-gguf-a.json").write_text("{not json")
    assert client.get("/v1/node/instance/gguf-a/status", headers=AUTH).json() == {"result": None}


def test_unauthenticated_is_401(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    assert client.post("/v1/node/instance/gguf-a", json={"verb": "create", "document": DOC}).status_code == 401
