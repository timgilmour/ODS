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
    profiles = r.json()["profiles"]
    assert len(profiles) == 2
    assert profiles[0] == {"name": "laguna", "engine": "vllm", "health_url": None, "container": None}
    assert profiles[1] == {"name": "mm27b", "engine": "vllm", "health_url": None, "container": None}
    assert r.json()["swap_status"] is None


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


def test_list_profiles_returns_dicts_with_default_engine(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, profiles=("heretic",))
    r = client.get("/v1/node/profiles", headers=AUTH)
    assert r.status_code == 200
    profiles = r.json()["profiles"]
    assert len(profiles) == 1
    assert profiles[0] == {
        "name": "heretic", "engine": "vllm", "health_url": None, "container": None
    }


def test_profiles_json_overrides_metadata(monkeypatch, tmp_path):
    vllm, ctl = _enable(monkeypatch, tmp_path, profiles=("comfyui",))
    (vllm / "profiles.json").write_text(json.dumps({
        "comfyui": {"engine": "comfyui",
                    "health_url": "http://127.0.0.1:8188/system_stats",
                    "container": "spark-comfyui"}}))
    r = client.get("/v1/node/profiles", headers=AUTH)
    assert r.status_code == 200
    profiles = r.json()["profiles"]
    comfyui_profile = next(p for p in profiles if p["name"] == "comfyui")
    assert comfyui_profile["engine"] == "comfyui"
    assert comfyui_profile["container"] == "spark-comfyui"


def test_malformed_profiles_json_falls_back_to_defaults(monkeypatch, tmp_path):
    vllm, ctl = _enable(monkeypatch, tmp_path, profiles=("heretic",))
    (vllm / "profiles.json").write_text("{not json")
    r = client.get("/v1/node/profiles", headers=AUTH)
    assert r.status_code == 200
    profiles = r.json()["profiles"]
    heretic = next(p for p in profiles if p["name"] == "heretic")
    assert heretic["engine"] == "vllm"


def test_profile_meta_tolerates_non_dict_entry_value(monkeypatch, tmp_path):
    """profiles.json is valid JSON but an entry's value isn't an object (e.g.
    {"comfyui": 5}). Regression for a 500 on GET /v1/node/serving: the old
    code did entry.get(key, default) unconditionally, and int has no .get."""
    vllm, ctl = _enable(monkeypatch, tmp_path, profiles=("comfyui",))
    (vllm / "profiles.json").write_text(json.dumps({"comfyui": 5}))

    meta = swapctl.profile_meta("comfyui")
    assert meta == {"name": "comfyui", "engine": "vllm",
                    "health_url": None, "container": None}

    # list_profiles() and current_profile_meta() route through profile_meta()
    # too -- neither may raise.
    profiles = swapctl.list_profiles()
    assert profiles == [meta]

    (ctl / "status.json").write_text(json.dumps(
        {"state": "done", "profile": "comfyui", "id": "x",
         "message": "swap launched", "ts": "2026-07-31T00:00:00Z"}))
    assert swapctl.current_profile_meta() == meta


def test_current_profile_meta_reads_status(monkeypatch, tmp_path):
    vllm, ctl = _enable(monkeypatch, tmp_path, profiles=("comfyui",))
    (vllm / "profiles.json").write_text(json.dumps(
        {"comfyui": {"engine": "comfyui"}}))
    (ctl / "status.json").write_text(json.dumps(
        {"state": "done", "profile": "comfyui", "id": "x",
         "message": "swap launched", "ts": "2026-07-31T00:00:00Z"}))
    meta = swapctl.current_profile_meta()
    assert meta is not None
    assert meta["engine"] == "comfyui"
