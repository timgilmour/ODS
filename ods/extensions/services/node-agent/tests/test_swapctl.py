"""/v1/node/profiles and /v1/node/swap — file-protocol swap control.

The agent never touches docker: POST /swap only writes request.json into the
shared ctl dir for the host-side swap-helper, and GET /profiles reads the
compose-*.yaml set (ro mount) plus the helper's status.json.
"""

import json

from fastapi.testclient import TestClient

import swapctl
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def _enable(monkeypatch, tmp_path, profiles=("mm27b", "laguna")):
    vllm = tmp_path / "vllm"
    ctl = tmp_path / "ctl"
    vllm.mkdir(exist_ok=True)
    ctl.mkdir(exist_ok=True)
    for p in profiles:
        (vllm / f"compose-{p}.yaml").write_text("services: {}\n")
    monkeypatch.setattr(swapctl.nodeconfig, "NODE_VLLM_DIR", str(vllm))
    monkeypatch.setattr(swapctl.nodeconfig, "NODE_SWAP_CTL_DIR", str(ctl))
    return vllm, ctl


def test_profiles_requires_auth():
    assert client.get("/v1/node/profiles").status_code == 401


def test_swap_requires_auth():
    assert client.post("/v1/node/swap", json={"profile": "x"}).status_code == 401


def test_profiles_disabled_when_unconfigured(monkeypatch):
    monkeypatch.setattr(swapctl.nodeconfig, "NODE_VLLM_DIR", "")
    monkeypatch.setattr(swapctl.nodeconfig, "NODE_SWAP_CTL_DIR", "")
    r = client.get("/v1/node/profiles", headers=AUTH)
    assert r.status_code == 503


def test_profiles_lists_compose_set_with_null_status(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    r = client.get("/v1/node/profiles", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"profiles": ["laguna", "mm27b"], "swap_status": None}


def test_profiles_surfaces_helper_status(monkeypatch, tmp_path):
    _, ctl = _enable(monkeypatch, tmp_path)
    status = {"state": "done", "profile": "laguna", "id": "abc",
              "message": "swap launched", "ts": "2026-07-30T22:00:00Z"}
    (ctl / "status.json").write_text(json.dumps(status))
    r = client.get("/v1/node/profiles", headers=AUTH)
    assert r.json()["swap_status"] == status


def test_swap_writes_request_and_returns_202(monkeypatch, tmp_path):
    _, ctl = _enable(monkeypatch, tmp_path)
    r = client.post("/v1/node/swap", json={"profile": "laguna"}, headers=AUTH)
    assert r.status_code == 202
    req = json.loads((ctl / "request.json").read_text())
    assert req["profile"] == "laguna"
    assert req["id"] == r.json()["id"]


def test_swap_unknown_profile_404_writes_nothing(monkeypatch, tmp_path):
    _, ctl = _enable(monkeypatch, tmp_path)
    r = client.post("/v1/node/swap", json={"profile": "ghost"}, headers=AUTH)
    assert r.status_code == 404
    assert not (ctl / "request.json").exists()


def test_swap_malformed_name_400(monkeypatch, tmp_path):
    _, ctl = _enable(monkeypatch, tmp_path)
    r = client.post("/v1/node/swap", json={"profile": "../evil"}, headers=AUTH)
    assert r.status_code == 400
    assert not (ctl / "request.json").exists()


def test_swap_conflicts_with_pending_request(monkeypatch, tmp_path):
    _, ctl = _enable(monkeypatch, tmp_path)
    (ctl / "request.json").write_text(json.dumps({"id": "x", "profile": "mm27b"}))
    r = client.post("/v1/node/swap", json={"profile": "laguna"}, headers=AUTH)
    assert r.status_code == 409


def test_swap_conflicts_while_helper_swapping(monkeypatch, tmp_path):
    _, ctl = _enable(monkeypatch, tmp_path)
    (ctl / "status.json").write_text(json.dumps(
        {"state": "swapping", "profile": "mm27b", "id": "x",
         "message": "swap.sh running", "ts": "2026-07-30T22:00:00Z"}))
    r = client.post("/v1/node/swap", json={"profile": "laguna"}, headers=AUTH)
    assert r.status_code == 409


def test_swap_disabled_when_unconfigured(monkeypatch):
    monkeypatch.setattr(swapctl.nodeconfig, "NODE_VLLM_DIR", "")
    monkeypatch.setattr(swapctl.nodeconfig, "NODE_SWAP_CTL_DIR", "")
    r = client.post("/v1/node/swap", json={"profile": "laguna"}, headers=AUTH)
    assert r.status_code == 503


def test_info_advertises_swap_capability_when_enabled(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    r = client.get("/v1/node/info", headers=AUTH)
    assert "swap" in r.json()["capabilities"]
