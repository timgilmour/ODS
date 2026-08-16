"""engines.json declaration + read-only engine status probe.

Mirrors swapctl.py's privilege split: this agent only ever READS
engines.json (host-owned, lives beside profiles.json) -- see swapctl's
module docstring. T3 adds the up/down request-file writer on top of this;
nothing here builds toward that.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import engines
from app import app
from engines import EngineDecl, count_established, engine_status, load_engines

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}

DECL = EngineDecl(
    name="omni",
    compose_file="/home/tim/omni/compose-omni.yaml",
    health_url="http://127.0.0.1:8008/health",
    busy_port=8008,
)


def _valid_entry(**overrides):
    entry = {
        "compose_file": "/home/tim/omni/compose-omni.yaml",
        "health_url": "http://127.0.0.1:8008/health",
        "busy": {"kind": "connections", "port": 8008},
    }
    entry.update(overrides)
    return entry


def _ok_client(status_code=200):
    """A minimal client stand-in with a real httpx.Response so
    raise_for_status() behaves exactly as it would for a real GET."""
    class _Client:
        def get(self, url, timeout=None):
            return httpx.Response(status_code, request=httpx.Request("GET", url))
    return _Client()


def _failing_client():
    class _Client:
        def get(self, url, timeout=None):
            raise httpx.ConnectError("connection refused",
                                     request=httpx.Request("GET", url))
    return _Client()


def _failing_counter(port):
    raise OSError("[Errno 13] Permission denied: '/proc/net/tcp'")


# ---------------------------------------------------------------------------
# load_engines: strict loader over engines.json.
# ---------------------------------------------------------------------------


def test_load_engines_missing_file_is_empty(tmp_path):
    assert load_engines(tmp_path / "nope.json") == {}


def test_load_engines_happy_path(tmp_path):
    p = tmp_path / "engines.json"
    p.write_text(json.dumps({"omni": _valid_entry()}))
    decls = load_engines(p)
    assert decls == {"omni": DECL}


def test_load_engines_rejects_relative_compose(tmp_path):
    p = tmp_path / "engines.json"
    p.write_text(json.dumps({"omni": _valid_entry(compose_file="omni/c.yaml")}))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_unknown_busy_kind(tmp_path):
    # {"kind": "metrics", ...} is refused today -- refuse-never-coerce, no
    # speculative kinds (returns only if upstream re-grows metrics support)
    p = tmp_path / "engines.json"
    p.write_text(json.dumps({"omni": _valid_entry(
        busy={"kind": "metrics", "port": 8008})}))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_extra_entry_key(tmp_path):
    p = tmp_path / "engines.json"
    entry = _valid_entry()
    entry["notes"] = "unexpected"
    p.write_text(json.dumps({"omni": entry}))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_missing_entry_key(tmp_path):
    p = tmp_path / "engines.json"
    entry = _valid_entry()
    del entry["health_url"]
    p.write_text(json.dumps({"omni": entry}))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_extra_busy_key(tmp_path):
    p = tmp_path / "engines.json"
    p.write_text(json.dumps({"omni": _valid_entry(
        busy={"kind": "connections", "port": 8008, "path": "/metrics"})}))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_port_out_of_range(tmp_path):
    p = tmp_path / "engines.json"
    p.write_text(json.dumps({"omni": _valid_entry(
        busy={"kind": "connections", "port": 70000})}))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_bool_port(tmp_path):
    """bool is a subclass of int in Python -- refuse-never-coerce means
    True/False must never silently pass as 1/0."""
    p = tmp_path / "engines.json"
    p.write_text(json.dumps({"omni": _valid_entry(
        busy={"kind": "connections", "port": True})}))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_non_object_json(tmp_path):
    p = tmp_path / "engines.json"
    p.write_text(json.dumps(["omni"]))
    with pytest.raises(ValueError):
        load_engines(p)


def test_load_engines_rejects_malformed_json(tmp_path):
    p = tmp_path / "engines.json"
    p.write_text("{not json")
    with pytest.raises(ValueError):
        load_engines(p)


# ---------------------------------------------------------------------------
# engine_status: count established connections BEFORE the health GET.
# ---------------------------------------------------------------------------


def test_status_unreachable_reports_busy_none():
    s = engine_status(DECL, count_established=_failing_counter,
                      client=_failing_client())
    assert s == {"reachable": False, "healthy": False, "busy_requests": None}


def test_status_healthy_counts_established_connections():
    s = engine_status(DECL, count_established=lambda port: 2,
                      client=_ok_client())
    assert s["healthy"] and s["busy_requests"] == 2


def test_status_counter_unavailable_is_none_not_zero():
    # counter raises; health 200 -> busy_requests None, NOT 0 (a caller
    # failing toward "busy" must never read an unknown count as idle)
    s = engine_status(DECL, count_established=_failing_counter,
                      client=_ok_client())
    assert s["healthy"] and s["busy_requests"] is None


def test_status_non_2xx_response_is_unhealthy():
    # health GET connects but returns a bad status -- raise_for_status()
    # must be exercised, not just "did the socket open". A genuine 0 count
    # also asserts the counter's own zero is never confused with the
    # None-on-failure sentinel from the tests above.
    s = engine_status(DECL, count_established=lambda port: 0,
                      client=_ok_client(status_code=500))
    assert s == {"reachable": False, "healthy": False, "busy_requests": 0}


def test_status_counts_before_health_probe():
    calls = []

    def counter(port):
        calls.append("count")
        return 0

    class _Client:
        def get(self, url, timeout=None):
            calls.append("health")
            return httpx.Response(200, request=httpx.Request("GET", url))

    engine_status(DECL, count_established=counter, client=_Client())
    assert calls == ["count", "health"]


# ---------------------------------------------------------------------------
# count_established: pure parser over /proc/net/tcp{,6} text.
# ---------------------------------------------------------------------------


_TCP_HEADER = ("  sl  local_address rem_address   st tx_queue rx_queue tr "
              "tm->when retrnsmt   uid  timeout inode")

# port 8008 == 0x1F48
_TCP_FIXTURE = "\n".join([
    _TCP_HEADER,
    "   0: 0100007F:1F48 00000000:0000 0A 00000000:00000000 00:00000000 "
    "00000000     0        0 12345 1 0000000000000000 100 0 0 10 0",
    "   1: 0100007F:1F48 0100007F:9C41 01 00000000:00000000 00:00000000 "
    "00000000     0        0 12346 1 0000000000000000 100 0 0 10 0",
    "   2: 0100007F:9C41 0100007F:1F48 01 00000000:00000000 00:00000000 "
    "00000000     0        0 12347 1 0000000000000000 100 0 0 10 0",
    "",
])

_TCP6_FIXTURE = "\n".join([
    _TCP_HEADER,
    "   0: 00000000000000000000000000000000:1F48 "
    "00000000000000000000000000000000:0000 0A 00000000:00000000 "
    "00:00000000 00000000     0        0 12348 1 "
    "0000000000000000 100 0 0 10 0",
    "   1: 00000000000000000000000000000001:1F48 "
    "00000000000000000000000000000001:9C41 01 00000000:00000000 "
    "00:00000000 00000000     0        0 12349 1 "
    "0000000000000000 100 0 0 10 0",
    "",
])


def test_count_established_parses_proc_net_tcp(tmp_path):
    # matches local_port==8008 hex AND st==01 (ESTABLISHED) across BOTH
    # files; fixture includes a LISTEN row (excluded by state) and a
    # foreign-port-8008 row (excluded because it's the FOREIGN port, not
    # local) that must NOT count.
    tcp = tmp_path / "tcp"
    tcp6 = tmp_path / "tcp6"
    tcp.write_text(_TCP_FIXTURE)
    tcp6.write_text(_TCP6_FIXTURE)
    assert count_established(8008, tcp_path=str(tcp), tcp6_path=str(tcp6)) == 2


def test_count_established_missing_proc_files_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        count_established(8008, tcp_path=str(tmp_path / "nope"),
                          tcp6_path=str(tmp_path / "nope6"))


# ---------------------------------------------------------------------------
# engines.json path resolution: NODE_ENGINES_FILE env, else
# <NODE_VLLM_DIR>/engines.json beside profiles.json, else unconfigured.
# ---------------------------------------------------------------------------


def test_configured_path_uses_env_override(monkeypatch, tmp_path):
    explicit = tmp_path / "custom-engines.json"
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", str(explicit))
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", str(tmp_path / "vllm"))
    assert engines._configured_path() == explicit


def test_configured_path_defaults_beside_profiles_json(monkeypatch, tmp_path):
    vllm = tmp_path / "vllm"
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", str(vllm))
    assert engines._configured_path() == vllm / "engines.json"


def test_configured_path_none_when_both_unset(monkeypatch):
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", "")
    assert engines._configured_path() is None


def test_load_configured_engines_empty_when_unconfigured(monkeypatch):
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", "")
    assert engines.load_configured_engines() == {}


# ---------------------------------------------------------------------------
# Routes: GET /v1/node/engines, GET /v1/node/engine/{name}/status.
# ---------------------------------------------------------------------------


def _configure(monkeypatch, tmp_path, entries):
    p = tmp_path / "engines.json"
    p.write_text(json.dumps(entries))
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", str(p))
    return p


def test_engines_requires_auth():
    assert client.get("/v1/node/engines").status_code == 401


def test_engine_status_requires_auth():
    assert client.get("/v1/node/engine/omni/status").status_code == 401


def test_list_engines_empty_when_unconfigured(monkeypatch):
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", "")
    r = client.get("/v1/node/engines", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"engines": []}


def test_list_engines_returns_names(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, {
        "omni": _valid_entry(),
        "aux": _valid_entry(health_url="http://127.0.0.1:9000/health",
                            busy={"kind": "connections", "port": 9000}),
    })
    r = client.get("/v1/node/engines", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"engines": ["aux", "omni"]}


def test_engine_status_unknown_is_404(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, {"omni": _valid_entry()})
    r = client.get("/v1/node/engine/ghost/status", headers=AUTH)
    assert r.status_code == 404


def test_engine_status_unknown_is_404_when_unconfigured(monkeypatch):
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", "")
    r = client.get("/v1/node/engine/omni/status", headers=AUTH)
    assert r.status_code == 404


def test_engine_status_known_returns_probe_result(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, {"omni": _valid_entry()})
    monkeypatch.setattr(engines, "engine_status", lambda decl: {
        "reachable": True, "healthy": True, "busy_requests": 3})
    r = client.get("/v1/node/engine/omni/status", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"reachable": True, "healthy": True, "busy_requests": 3}


# ---------------------------------------------------------------------------
# Routes: POST /v1/node/engine/{name}/up, /down -- write engine-req.json for
# the host-side swap-helper (Task 2) to consume. The agent has no docker
# access; writing the request file is the entire actuation, and the agent
# never waits for or reads engine-status-<resource>.json (forensics only).
# ---------------------------------------------------------------------------


def _enable_engine_ctl(monkeypatch, tmp_path, entries=None):
    """Configure engines.json AND the swap-ctl dirs the writer needs
    (swapctl._dirs() -- NODE_VLLM_DIR/NODE_SWAP_CTL_DIR), mirroring
    test_swapctl.py's _enable() alongside this file's _configure()."""
    vllm = tmp_path / "vllm"
    ctl = tmp_path / "ctl"
    vllm.mkdir(exist_ok=True)
    ctl.mkdir(exist_ok=True)
    p = vllm / "engines.json"
    p.write_text(json.dumps(entries if entries is not None else {"omni": _valid_entry()}))
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", str(vllm))
    monkeypatch.setattr(engines.swapctl.nodeconfig, "NODE_SWAP_CTL_DIR", str(ctl))
    return vllm, ctl


def test_engine_up_requires_auth():
    assert client.post("/v1/node/engine/omni/up").status_code == 401


def test_engine_down_requires_auth():
    assert client.post("/v1/node/engine/omni/down").status_code == 401


def test_engine_up_writes_request_and_returns_202(monkeypatch, tmp_path):
    _, ctl = _enable_engine_ctl(monkeypatch, tmp_path)
    r = client.post("/v1/node/engine/omni/up", headers=AUTH)
    assert r.status_code == 202
    assert r.json() == {"accepted": True}
    req = json.loads((ctl / "engine-req.json").read_text())
    assert req["resource"] == "omni"
    assert req["verb"] == "up"
    assert isinstance(req["ts"], float)


def test_engine_down_writes_request_and_returns_202(monkeypatch, tmp_path):
    _, ctl = _enable_engine_ctl(monkeypatch, tmp_path)
    r = client.post("/v1/node/engine/omni/down", headers=AUTH)
    assert r.status_code == 202
    assert r.json() == {"accepted": True}
    req = json.loads((ctl / "engine-req.json").read_text())
    assert req["resource"] == "omni"
    assert req["verb"] == "down"


def test_engine_up_request_written_atomically(monkeypatch, tmp_path):
    # No stray .tmp files left behind once the request lands.
    _, ctl = _enable_engine_ctl(monkeypatch, tmp_path)
    client.post("/v1/node/engine/omni/up", headers=AUTH)
    leftovers = [p for p in ctl.iterdir() if p.name != "engine-req.json"]
    assert leftovers == []


def test_engine_up_unknown_engine_404(monkeypatch, tmp_path):
    _, ctl = _enable_engine_ctl(monkeypatch, tmp_path)
    r = client.post("/v1/node/engine/ghost/up", headers=AUTH)
    assert r.status_code == 404
    assert not (ctl / "engine-req.json").exists()


def test_engine_down_unknown_engine_404(monkeypatch, tmp_path):
    _, ctl = _enable_engine_ctl(monkeypatch, tmp_path)
    r = client.post("/v1/node/engine/ghost/down", headers=AUTH)
    assert r.status_code == 404
    assert not (ctl / "engine-req.json").exists()


def test_engine_up_unknown_engine_404_when_unconfigured(monkeypatch):
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", "")
    r = client.post("/v1/node/engine/omni/up", headers=AUTH)
    assert r.status_code == 404


def test_engine_up_conflicts_with_pending_request(monkeypatch, tmp_path):
    _, ctl = _enable_engine_ctl(monkeypatch, tmp_path)
    (ctl / "engine-req.json").write_text(
        json.dumps({"resource": "omni", "verb": "up", "ts": 1.0}))
    r = client.post("/v1/node/engine/omni/up", headers=AUTH)
    assert r.status_code == 409


def test_engine_down_conflicts_with_pending_request(monkeypatch, tmp_path):
    # A pending request of any verb blocks a new one -- the file is a
    # one-slot queue, not keyed per-resource or per-verb.
    _, ctl = _enable_engine_ctl(monkeypatch, tmp_path)
    (ctl / "engine-req.json").write_text(
        json.dumps({"resource": "omni", "verb": "up", "ts": 1.0}))
    r = client.post("/v1/node/engine/omni/down", headers=AUTH)
    assert r.status_code == 409


def test_engine_up_disabled_when_swap_ctl_unconfigured(monkeypatch, tmp_path):
    # engines.json IS configured/declared, but NODE_SWAP_CTL_DIR is not --
    # the agent has nowhere to write the request file.
    vllm = tmp_path / "vllm"
    vllm.mkdir(exist_ok=True)
    (vllm / "engines.json").write_text(json.dumps({"omni": _valid_entry()}))
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", str(vllm))
    monkeypatch.setattr(engines.swapctl.nodeconfig, "NODE_SWAP_CTL_DIR", "")
    r = client.post("/v1/node/engine/omni/up", headers=AUTH)
    assert r.status_code == 503


def test_engine_down_disabled_when_swap_ctl_unconfigured(monkeypatch, tmp_path):
    vllm = tmp_path / "vllm"
    vllm.mkdir(exist_ok=True)
    (vllm / "engines.json").write_text(json.dumps({"omni": _valid_entry()}))
    monkeypatch.setattr(engines.nodeconfig, "NODE_ENGINES_FILE", "")
    monkeypatch.setattr(engines.nodeconfig, "NODE_VLLM_DIR", str(vllm))
    monkeypatch.setattr(engines.swapctl.nodeconfig, "NODE_SWAP_CTL_DIR", "")
    r = client.post("/v1/node/engine/omni/down", headers=AUTH)
    assert r.status_code == 503
