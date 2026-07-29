# Multi-Node GPU Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remote inference nodes (first: DGX Spark "sparky") appear read-only on the ODS GPU page via a slim node-agent + dashboard-api registry/poller.

**Architecture:** New `node-agent` FastAPI service on the remote box speaks a versioned `/v1/node/*` protocol (vendoring dashboard-api's `gpu.py` collectors at build time). dashboard-api gains an `ODS_REMOTE_NODES` registry + background poller and an **additive** `nodes` field on `MultiGPUStatus`. `GPUMonitor.jsx` renders one section per node below the untouched local layout.

**Tech Stack:** Python 3.11 / FastAPI / httpx / pydantic (same as dashboard-api), React + vitest (dashboard), Docker.

**Spec:** `docs/superpowers/specs/2026-07-29-multi-node-gpu-visibility-design.md`

## Global Constraints

- Branch: `feat/remote-node-gpu-visibility` (already exists, tracks `origin/main`).
- Feature fully dormant when `ODS_REMOTE_NODES` is unset/empty — zero behavior change.
- `MultiGPUStatus` change is additive only; every existing field byte-for-byte unchanged.
- Remote nodes NEVER appear in deck topology/assignments.
- Node-agent default port **7720** (host-agent owns 7710). Auth: static bearer `NODE_AGENT_KEY`; unauthenticated → 401 with no detail body.
- Terminology caution: upstream already uses "node" for the *local* snapshot (`routers/node.py`) and "remote provider"/"peer" for inference routing. Our feature is **remote nodes (monitoring)**; do not touch remote-provider code.
- No new Python deps beyond what dashboard-api already uses (fastapi, httpx, pydantic, uvicorn).
- TDD every task; run the relevant suite before each commit. dashboard-api tests run: `cd ods/extensions/services/dashboard-api && python -m pytest tests/ -x -q` (conftest puts the service dir on sys.path). Dashboard tests: `cd ods/extensions/services/dashboard && npx vitest run`.
- Work from repo root `~/projects/ODS` unless a step says otherwise.

## Reference shapes (from the existing codebase — do not redefine elsewhere)

`ods/extensions/services/dashboard-api/models.py:129`:

```python
class IndividualGPU(BaseModel):
    index: int
    uuid: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    utilization_percent: int
    temperature_c: int
    power_w: Optional[float] = None
    memory_type: str = "discrete"
    assigned_services: list[str] = []
    memory_usage_available: bool = True
    utilization_available: bool = True
    temperature_available: bool = True
```

Collector entrypoints in `dashboard-api/gpu.py`: `get_gpu_info_nvidia_detailed() -> Optional[list[IndividualGPU]]` (line ~619), `get_gpu_info_amd_detailed()` (~718). `gpu.py` imports only stdlib + `models` (GPUInfo, IndividualGPU) + `host_agent_client` (AgentClientError, request_json) — the node-agent shims those two modules.

`_lifespan` in `dashboard-api/main.py:1032` starts background tasks with `asyncio.create_task(...)`.

---

### Task 1: node-agent scaffolding, auth, `/v1/node/info`

**Files:**
- Create: `ods/extensions/services/node-agent/app.py`
- Create: `ods/extensions/services/node-agent/nodeconfig.py`
- Create: `ods/extensions/services/node-agent/models.py` (shim)
- Create: `ods/extensions/services/node-agent/host_agent_client.py` (shim)
- Create: `ods/extensions/services/node-agent/requirements.txt`
- Create: `ods/extensions/services/node-agent/tests/conftest.py`
- Test: `ods/extensions/services/node-agent/tests/test_info.py`

**Interfaces:**
- Produces: FastAPI `app` in `app.py`; `verify_key` dependency; `GET /v1/node/info` → `{"name", "hostname", "platform", "capabilities": ["metrics"], "gpus": [...]}`; env config in `nodeconfig.py`: `NODE_AGENT_KEY`, `NODE_NAME`, `GPU_BACKEND` (default `"nvidia"`), `NODE_SERVING_PROBE_URL`, `NODE_SERVING_CONTAINER`, `NODE_AGENT_PORT` (default 7720).
- Consumes: vendored `gpu.py` (copied in Task 4's Docker build; for dev/tests, conftest sys-path-inserts `../dashboard-api` is NOT used — the node-agent dir must be self-sufficient, so tests monkeypatch the collector).

- [ ] **Step 1: Write shims and conftest**

`models.py` (shim — exact copy of the two classes from dashboard-api, nothing else):

```python
"""Shim satisfying gpu.py imports. Field parity with dashboard-api/models.py
is enforced by tests/test_model_parity.py."""
from typing import Optional
from pydantic import BaseModel


class GPUInfo(BaseModel):
    name: str
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    utilization_percent: int
    temperature_c: int
    power_w: Optional[float] = None
    memory_type: str = "discrete"
    gpu_backend: str = "nvidia"
    gpu_count: int = 1
    memory_usage_available: bool = True
    utilization_available: bool = True
    temperature_available: bool = True


class IndividualGPU(BaseModel):
    index: int
    uuid: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    utilization_percent: int
    temperature_c: int
    power_w: Optional[float] = None
    memory_type: str = "discrete"
    assigned_services: list[str] = []
    memory_usage_available: bool = True
    utilization_available: bool = True
    temperature_available: bool = True
```

`host_agent_client.py` (shim — there is no host agent on a remote node):

```python
"""Shim satisfying gpu.py imports; windows-host paths are unsupported here."""


class AgentClientError(RuntimeError):
    pass


def request_json(*args, **kwargs):
    raise AgentClientError("host-agent not available on a remote node")
```

`nodeconfig.py`:

```python
import os
import socket

NODE_AGENT_KEY = os.environ.get("NODE_AGENT_KEY", "")
NODE_NAME = os.environ.get("NODE_NAME", socket.gethostname())
GPU_BACKEND = os.environ.get("GPU_BACKEND", "nvidia").lower()
NODE_SERVING_PROBE_URL = os.environ.get("NODE_SERVING_PROBE_URL", "")
NODE_SERVING_CONTAINER = os.environ.get("NODE_SERVING_CONTAINER", "")
NODE_AGENT_PORT = int(os.environ.get("NODE_AGENT_PORT", "7720"))
GPU_CACHE_TTL_SECONDS = float(os.environ.get("NODE_GPU_CACHE_TTL", "2.0"))
```

`requirements.txt`:

```
fastapi>=0.111
uvicorn>=0.30
httpx>=0.27
pydantic>=2.7
```

`tests/conftest.py`:

```python
import os
import sys
from pathlib import Path

os.environ.setdefault("NODE_AGENT_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 2: Write the failing test**

`tests/test_info.py`:

```python
from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def _fake_gpus():
    return [{
        "index": 0, "uuid": "GPU-abc", "name": "NVIDIA GB10",
        "memory_used_mb": 1024, "memory_total_mb": 122880,
        "memory_percent": 0.8, "utilization_percent": 5,
        "temperature_c": 45, "power_w": 30.0, "memory_type": "unified",
    }]


def test_info_requires_auth():
    assert client.get("/v1/node/info").status_code == 401
    bad = client.get("/v1/node/info", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401


def test_info_returns_identity_and_inventory(monkeypatch):
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake_gpus)
    r = client.get("/v1/node/info", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["capabilities"] == ["metrics"]
    assert body["platform"] == "nvidia"
    assert body["gpus"][0]["name"] == "NVIDIA GB10"
    assert body["name"]
    assert body["hostname"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/test_info.py -q`
Expected: FAIL (`ModuleNotFoundError: app` or attribute errors)

- [ ] **Step 4: Write minimal `app.py`**

```python
"""ODS node-agent: read-only metrics endpoint for remote inference nodes."""
import socket
import time

from fastapi import Depends, FastAPI, Header, HTTPException

import nodeconfig
from gpu_collect import collect_detailed_gpus

app = FastAPI(title="ods-node-agent")


def verify_key(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {nodeconfig.NODE_AGENT_KEY}"
    if not nodeconfig.NODE_AGENT_KEY or authorization != expected:
        raise HTTPException(status_code=401)


def _collect_gpus_uncached() -> list[dict]:
    gpus = collect_detailed_gpus(nodeconfig.GPU_BACKEND)
    return [g.model_dump() for g in (gpus or [])]


_gpu_cache: dict = {"expires": 0.0, "value": None}


def _collect_gpus_cached() -> list[dict]:
    now = time.monotonic()
    if now < _gpu_cache["expires"] and _gpu_cache["value"] is not None:
        return _gpu_cache["value"]
    value = _collect_gpus_uncached()
    _gpu_cache["expires"] = now + nodeconfig.GPU_CACHE_TTL_SECONDS
    _gpu_cache["value"] = value
    return value


@app.get("/v1/node/info", dependencies=[Depends(verify_key)])
def node_info():
    return {
        "name": nodeconfig.NODE_NAME,
        "hostname": socket.gethostname(),
        "platform": nodeconfig.GPU_BACKEND,
        "capabilities": ["metrics"],
        "gpus": _collect_gpus_uncached(),
    }
```

And `gpu_collect.py` (thin selector over the vendored collectors; import guarded so unit tests never need real `gpu.py`):

```python
"""Selects the vendored dashboard-api collector for this node's backend."""
from typing import Optional

try:
    from gpu import get_gpu_info_amd_detailed, get_gpu_info_nvidia_detailed
except ImportError:  # pragma: no cover - vendored at container build time
    get_gpu_info_nvidia_detailed = None
    get_gpu_info_amd_detailed = None


def collect_detailed_gpus(backend: str) -> Optional[list]:
    if backend == "amd" and get_gpu_info_amd_detailed:
        return get_gpu_info_amd_detailed()
    if get_gpu_info_nvidia_detailed:
        return get_gpu_info_nvidia_detailed()
    return None
```

(Also add `gpu_collect.py` to the Files list of this task: Create.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/ -q`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add ods/extensions/services/node-agent
git commit -m "feat(node-agent): scaffold service with auth and /v1/node/info"
```

---

### Task 2: node-agent `/v1/node/gpu` with TTL cache

**Files:**
- Modify: `ods/extensions/services/node-agent/app.py`
- Test: `ods/extensions/services/node-agent/tests/test_gpu.py`

**Interfaces:**
- Produces: `GET /v1/node/gpu` → `{"backend": "<nvidia|amd>", "gpus": [IndividualGPU-shaped dicts]}`, TTL-cached ~2s so dashboard polling can't spam nvidia-smi.

- [ ] **Step 1: Write the failing test**

`tests/test_gpu.py`:

```python
from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def _fake(name):
    def _inner():
        return [{
            "index": 0, "uuid": "GPU-abc", "name": name,
            "memory_used_mb": 2048, "memory_total_mb": 122880,
            "memory_percent": 1.7, "utilization_percent": 40,
            "temperature_c": 55, "power_w": 90.0, "memory_type": "unified",
        }]
    return _inner


def test_gpu_requires_auth():
    assert client.get("/v1/node/gpu").status_code == 401


def test_gpu_returns_metrics(monkeypatch):
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("NVIDIA GB10"))
    r = client.get("/v1/node/gpu", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "nvidia"
    assert body["gpus"][0]["utilization_percent"] == 40


def test_gpu_uses_ttl_cache(monkeypatch):
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("FIRST"))
    assert client.get("/v1/node/gpu", headers=AUTH).json()["gpus"][0]["name"] == "FIRST"
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("SECOND"))
    # within TTL: still FIRST
    assert client.get("/v1/node/gpu", headers=AUTH).json()["gpus"][0]["name"] == "FIRST"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/test_gpu.py -q`
Expected: FAIL (404 — route missing)

- [ ] **Step 3: Implement the route in `app.py`**

```python
@app.get("/v1/node/gpu", dependencies=[Depends(verify_key)])
def node_gpu():
    return {"backend": nodeconfig.GPU_BACKEND, "gpus": _collect_gpus_cached()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ods/extensions/services/node-agent
git commit -m "feat(node-agent): add /v1/node/gpu with TTL cache"
```

---

### Task 3: node-agent `/v1/node/serving`

**Files:**
- Create: `ods/extensions/services/node-agent/serving.py`
- Modify: `ods/extensions/services/node-agent/app.py`
- Test: `ods/extensions/services/node-agent/tests/test_serving.py`

**Interfaces:**
- Produces: `GET /v1/node/serving` → `{"model": str|null, "endpoint_ok": bool, "container_status": str|null}`. `serving.probe()` uses `httpx` GET on `NODE_SERVING_PROBE_URL` (2s timeout, first model id from OpenAI `/v1/models` shape) and `docker inspect` for `NODE_SERVING_CONTAINER` when set (docker CLI + socket are optional; absence → `container_status: null`).

- [ ] **Step 1: Write the failing test**

`tests/test_serving.py`:

```python
from fastapi.testclient import TestClient

import serving
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def test_serving_requires_auth():
    assert client.get("/v1/node/serving").status_code == 401


def test_probe_happy_path(monkeypatch):
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": [{"id": "heretic"}]})
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "aeon-vllm")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json() == {"model": "heretic", "endpoint_ok": True,
                        "container_status": "running"}


def test_probe_endpoint_down(monkeypatch):
    def _boom(url):
        raise serving.ProbeError("connect timeout")
    monkeypatch.setattr(serving, "_fetch_models_payload", _boom)
    monkeypatch.setattr(serving, "_container_status", lambda name: None)
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    r = client.get("/v1/node/serving", headers=AUTH)
    body = r.json()
    assert body["endpoint_ok"] is False
    assert body["model"] is None


def test_probe_unconfigured(monkeypatch):
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json() == {"model": None, "endpoint_ok": False,
                        "container_status": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/test_serving.py -q`
Expected: FAIL (`ModuleNotFoundError: serving`)

- [ ] **Step 3: Implement `serving.py` and wire the route**

`serving.py`:

```python
"""Probe what this node is serving (OpenAI-compatible endpoint + container)."""
import subprocess

import httpx

import nodeconfig


class ProbeError(RuntimeError):
    pass


def _fetch_models_payload(url: str) -> dict:
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ProbeError(str(exc)) from exc


def _container_status(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True, text=True, timeout=2.0,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def probe() -> dict:
    result = {"model": None, "endpoint_ok": False, "container_status": None}
    if nodeconfig.NODE_SERVING_CONTAINER:
        result["container_status"] = _container_status(
            nodeconfig.NODE_SERVING_CONTAINER)
    if not nodeconfig.NODE_SERVING_PROBE_URL:
        return result
    try:
        payload = _fetch_models_payload(nodeconfig.NODE_SERVING_PROBE_URL)
        models = payload.get("data") or []
        result["endpoint_ok"] = True
        if models:
            result["model"] = models[0].get("id")
    except ProbeError:
        pass
    return result
```

In `app.py`:

```python
import serving

@app.get("/v1/node/serving", dependencies=[Depends(verify_key)])
def node_serving():
    return serving.probe()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ods/extensions/services/node-agent
git commit -m "feat(node-agent): add /v1/node/serving probe"
```

---

### Task 4: model-parity guard, Dockerfile, compose, README

**Files:**
- Test: `ods/extensions/services/node-agent/tests/test_model_parity.py`
- Create: `ods/extensions/services/node-agent/Dockerfile`
- Create: `ods/extensions/services/node-agent/compose.yaml`
- Create: `ods/extensions/services/node-agent/README.md`

**Interfaces:**
- Produces: drift guard between shim models and dashboard-api models; deployable container image (repo-root build context vendors `dashboard-api/gpu.py`).

- [ ] **Step 1: Write the failing-if-drifted parity test**

`tests/test_model_parity.py`:

```python
"""Guards shim model parity with dashboard-api. Runs in the repo checkout
(where both services exist); skipped inside the deployed container."""
import importlib.util
import sys
from pathlib import Path

import pytest

DASHBOARD_API = Path(__file__).resolve().parents[2] / "dashboard-api"


def _load_dashboard_models():
    if not (DASHBOARD_API / "models.py").exists():
        pytest.skip("dashboard-api not present (deployed container)")
    sys.path.insert(0, str(DASHBOARD_API))
    try:
        spec = importlib.util.spec_from_file_location(
            "da_models", DASHBOARD_API / "models.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def test_individual_gpu_fields_match():
    import models as shim
    da = _load_dashboard_models()
    assert set(shim.IndividualGPU.model_fields) == set(
        da.IndividualGPU.model_fields)


def test_gpu_info_fields_match():
    import models as shim
    da = _load_dashboard_models()
    assert set(shim.GPUInfo.model_fields) == set(da.GPUInfo.model_fields)
```

- [ ] **Step 2: Run it**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/test_model_parity.py -q`
Expected: PASS (or FAIL listing drifted fields — fix the shim to match, never the reverse). Note: dashboard-api/models.py imports `config`/`context_policy`; if module-level import fails, the loader executes in dashboard-api sys.path context which provides both — the `sys.path.insert` handles it.

- [ ] **Step 3: Write Dockerfile (repo-root context) and compose**

`Dockerfile`:

```dockerfile
# Build from repo root:  docker build -f ods/extensions/services/node-agent/Dockerfile .
FROM python:3.12-slim
WORKDIR /app
COPY ods/extensions/services/node-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Vendor the single-source GPU collectors from dashboard-api
COPY ods/extensions/services/dashboard-api/gpu.py ./gpu.py
COPY ods/extensions/services/node-agent/*.py ./
EXPOSE 7720
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${NODE_AGENT_PORT:-7720}"]
```

`compose.yaml`:

```yaml
# Runs on the REMOTE node. Build from a repo checkout root:
#   docker compose -f ods/extensions/services/node-agent/compose.yaml up -d --build
services:
  ods-node-agent:
    build:
      context: ../../../..
      dockerfile: ods/extensions/services/node-agent/Dockerfile
    container_name: ods-node-agent
    restart: unless-stopped
    network_mode: host
    pid: host
    environment:
      NODE_AGENT_KEY: ${NODE_AGENT_KEY:?set NODE_AGENT_KEY}
      NODE_NAME: ${NODE_NAME:-}
      GPU_BACKEND: ${GPU_BACKEND:-nvidia}
      NODE_SERVING_PROBE_URL: ${NODE_SERVING_PROBE_URL:-}
      NODE_SERVING_CONTAINER: ${NODE_SERVING_CONTAINER:-}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /usr/bin/docker:/usr/bin/docker:ro
```

Note: nvidia-smi inside a slim container — the NVIDIA runtime injects it with `--gpus`; compose on DGX OS: add

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, utility]
```

(`utility` capability provides nvidia-smi.) Include this block in `compose.yaml`.

`README.md`: purpose, env table (the six vars), deploy command above, firewall note ("scope the port to your dashboard host, e.g. `ufw allow from <dashboard-ip> to any port 7720 proto tcp`"), and the protocol's three endpoints with example curl.

- [ ] **Step 4: Run full node-agent suite**

Run: `cd ods/extensions/services/node-agent && python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ods/extensions/services/node-agent
git commit -m "feat(node-agent): parity guard, Dockerfile, compose, README"
```

---

### Task 5: dashboard-api models — `RemoteNodeStatus` + additive `MultiGPUStatus.nodes` + back-compat contract

**Files:**
- Modify: `ods/extensions/services/dashboard-api/models.py` (after `IndividualGPU`, before `MultiGPUStatus`)
- Test: `ods/extensions/services/dashboard-api/tests/test_remote_nodes_models.py`

**Interfaces:**
- Produces: `RemoteNodeServing(model, endpoint_ok, container_status)`, `RemoteNodeStatus(name, display_name, platform, status, last_seen, gpus, serving, error)`, `MultiGPUStatus.nodes: list[RemoteNodeStatus] = []`. Consumed by Tasks 6–8.

- [ ] **Step 1: Write the failing test**

`tests/test_remote_nodes_models.py`:

```python
"""RemoteNodeStatus models + MultiGPUStatus back-compat contract."""
from models import GPUInfo, IndividualGPU, MultiGPUStatus, RemoteNodeServing, RemoteNodeStatus

BASELINE_MULTI_GPU_FIELDS = {
    "gpu_count", "backend", "gpus", "topology", "assignment",
    "split_mode", "tensor_split", "aggregate",
}


def _aggregate():
    return GPUInfo(name="agg", memory_used_mb=1, memory_total_mb=2,
                   memory_percent=50.0, utilization_percent=1, temperature_c=40)


def test_multigpustatus_backcompat_additive_only():
    fields = set(MultiGPUStatus.model_fields)
    assert BASELINE_MULTI_GPU_FIELDS <= fields
    assert fields - BASELINE_MULTI_GPU_FIELDS == {"nodes"}


def test_nodes_default_empty_and_serializes():
    status = MultiGPUStatus(gpu_count=0, backend="amd", gpus=[],
                            aggregate=_aggregate())
    assert status.nodes == []
    assert status.model_dump()["nodes"] == []


def test_remote_node_status_shape():
    node = RemoteNodeStatus(
        name="sparky", display_name="DGX Spark GB10", platform="nvidia",
        status="online", last_seen="2026-07-29T18:00:00+00:00",
        gpus=[IndividualGPU(index=0, uuid="GPU-x", name="GB10",
                            memory_used_mb=1, memory_total_mb=2,
                            memory_percent=50.0, utilization_percent=1,
                            temperature_c=40)],
        serving=RemoteNodeServing(model="heretic", endpoint_ok=True),
    )
    assert node.status == "online"
    assert node.error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ods/extensions/services/dashboard-api && python -m pytest tests/test_remote_nodes_models.py -q`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add models in `models.py`**

```python
class RemoteNodeServing(BaseModel):
    model: Optional[str] = None
    endpoint_ok: bool = False
    container_status: Optional[str] = None


class RemoteNodeStatus(BaseModel):
    name: str
    display_name: Optional[str] = None
    platform: str = "unknown"
    status: str  # "online" | "offline" | "error"
    last_seen: Optional[str] = None
    gpus: list[IndividualGPU] = []
    serving: Optional[RemoteNodeServing] = None
    error: Optional[str] = None
```

And on `MultiGPUStatus` add the last field: `nodes: list[RemoteNodeStatus] = []`.

- [ ] **Step 4: Run the full dashboard-api suite (back-compat proof)**

Run: `cd ods/extensions/services/dashboard-api && python -m pytest tests/ -x -q`
Expected: ALL PASS (existing tests untouched)

- [ ] **Step 5: Commit**

```bash
git add ods/extensions/services/dashboard-api/models.py ods/extensions/services/dashboard-api/tests/test_remote_nodes_models.py
git commit -m "feat(dashboard-api): remote node models, additive MultiGPUStatus.nodes"
```

---

### Task 6: dashboard-api `remote_nodes.py` — registry + poller

**Files:**
- Create: `ods/extensions/services/dashboard-api/remote_nodes.py`
- Test: `ods/extensions/services/dashboard-api/tests/test_remote_nodes_poller.py`

**Interfaces:**
- Produces: `load_remote_nodes() -> list[RemoteNodeConfig]` (env `ODS_REMOTE_NODES` JSON; malformed → log + `[]`); `get_remote_node_statuses() -> list[RemoteNodeStatus]`; `async poll_remote_nodes_forever(interval=5.0)`; `async poll_all_nodes_once(client)` (used by tests and the loop). Consumed by Task 7.

- [ ] **Step 1: Write the failing test**

`tests/test_remote_nodes_poller.py`:

```python
import asyncio
import json

import httpx
import pytest

import remote_nodes


NODES_ENV = json.dumps([{"name": "sparky", "display_name": "DGX Spark GB10",
                         "url": "http://sparky.test:7720",
                         "key_env": "TEST_NODE_KEY"}])

GPU = {"index": 0, "uuid": "GPU-x", "name": "GB10", "memory_used_mb": 1,
       "memory_total_mb": 2, "memory_percent": 50.0,
       "utilization_percent": 7, "temperature_c": 40}


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             timeout=2.0)


def setup_function(_fn):
    remote_nodes._STATE.clear()


def test_load_nodes_parses_env(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")
    nodes = remote_nodes.load_remote_nodes()
    assert len(nodes) == 1
    assert nodes[0].name == "sparky"
    assert nodes[0].key == "sekrit"


def test_load_nodes_malformed_returns_empty(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", "{not json")
    assert remote_nodes.load_remote_nodes() == []


def test_load_nodes_absent_returns_empty(monkeypatch):
    monkeypatch.delenv("ODS_REMOTE_NODES", raising=False)
    assert remote_nodes.load_remote_nodes() == []


@pytest.mark.asyncio
async def test_poll_online(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        assert request.headers["Authorization"] == "Bearer sekrit"
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia",
                                             "gpus": [GPU]})
        if request.url.path == "/v1/node/serving":
            return httpx.Response(200, json={"model": "heretic",
                                             "endpoint_ok": True,
                                             "container_status": "running"})
        return httpx.Response(404)

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "online"
    assert status.platform == "nvidia"
    assert status.gpus[0].utilization_percent == 7
    assert status.serving.model == "heretic"
    assert status.last_seen is not None


@pytest.mark.asyncio
async def test_poll_offline_preserves_last_seen(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def up(request):
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia", "gpus": [GPU]})
        return httpx.Response(200, json={"model": None, "endpoint_ok": False,
                                         "container_status": None})

    async with _client(up) as client:
        await remote_nodes.poll_all_nodes_once(client)
    seen = remote_nodes.get_remote_node_statuses()[0].last_seen

    def down(request):
        raise httpx.ConnectError("refused")

    async with _client(down) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "offline"
    assert status.last_seen == seen
    assert status.gpus == []


@pytest.mark.asyncio
async def test_poll_auth_failure_is_error_not_offline(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        return httpx.Response(401)

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "error"
    assert "401" in status.error


@pytest.mark.asyncio
async def test_one_bad_node_does_not_block_others(monkeypatch):
    two = json.loads(NODES_ENV) + [{"name": "deadbox",
                                    "url": "http://dead.test:7720",
                                    "key_env": "TEST_NODE_KEY"}]
    monkeypatch.setenv("ODS_REMOTE_NODES", json.dumps(two))
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        if request.url.host == "dead.test":
            raise httpx.ConnectError("refused")
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia", "gpus": [GPU]})
        return httpx.Response(200, json={"model": None, "endpoint_ok": False,
                                         "container_status": None})

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    by_name = {s.name: s for s in remote_nodes.get_remote_node_statuses()}
    assert by_name["sparky"].status == "online"
    assert by_name["deadbox"].status == "offline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ods/extensions/services/dashboard-api && python -m pytest tests/test_remote_nodes_poller.py -q`
Expected: FAIL (ImportError). If `pytest.mark.asyncio` is unknown, check `tests/requirements-test.txt` for the async plugin the suite already uses (`pytest-asyncio` or `anyio`) and match its idiom — several existing tests are async; copy their marker style.

- [ ] **Step 3: Implement `remote_nodes.py`**

```python
"""Remote inference node registry + read-only metrics poller.

Nodes are configured via ODS_REMOTE_NODES (JSON list of
{"name", "display_name"?, "url", "key_env"}). Keys are env-var *names*,
never inline secrets. Absent/empty config → feature dormant.
Terminology: distinct from routers/node.py (local snapshot) and from the
remote-provider/peer inference-routing machinery.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from models import IndividualGPU, RemoteNodeServing, RemoteNodeStatus

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0
NODE_TIMEOUT_SECONDS = 2.0

_STATE: dict[str, RemoteNodeStatus] = {}


@dataclass(frozen=True)
class RemoteNodeConfig:
    name: str
    url: str
    key: str
    display_name: str | None = None


def load_remote_nodes() -> list[RemoteNodeConfig]:
    raw = os.environ.get("ODS_REMOTE_NODES", "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
        assert isinstance(entries, list)
    except (ValueError, AssertionError):
        logger.warning("ODS_REMOTE_NODES is not a JSON list; ignoring")
        return []
    nodes: list[RemoteNodeConfig] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") \
                or not entry.get("url"):
            logger.warning("Skipping malformed remote node entry: %r", entry)
            continue
        key = os.environ.get(entry.get("key_env", ""), "")
        nodes.append(RemoteNodeConfig(
            name=str(entry["name"]), url=str(entry["url"]).rstrip("/"),
            key=key, display_name=entry.get("display_name")))
    return nodes


def get_remote_node_statuses() -> list[RemoteNodeStatus]:
    return [_STATE[cfg.name] for cfg in load_remote_nodes()
            if cfg.name in _STATE]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _carry(cfg: RemoteNodeConfig, status: str, error: str | None) -> RemoteNodeStatus:
    prev = _STATE.get(cfg.name)
    return RemoteNodeStatus(
        name=cfg.name, display_name=cfg.display_name,
        platform=prev.platform if prev else "unknown",
        status=status, last_seen=prev.last_seen if prev else None,
        gpus=[], serving=None, error=error)


async def _poll_node_once(cfg: RemoteNodeConfig,
                          client: httpx.AsyncClient) -> RemoteNodeStatus:
    headers = {"Authorization": f"Bearer {cfg.key}"}
    try:
        gpu_resp = await client.get(f"{cfg.url}/v1/node/gpu", headers=headers)
        serving_resp = await client.get(f"{cfg.url}/v1/node/serving",
                                        headers=headers)
        if gpu_resp.status_code != 200:
            return _carry(cfg, "error",
                          f"node returned HTTP {gpu_resp.status_code}")
        gpu_body = gpu_resp.json()
        serving = None
        if serving_resp.status_code == 200:
            serving = RemoteNodeServing(**serving_resp.json())
        return RemoteNodeStatus(
            name=cfg.name, display_name=cfg.display_name,
            platform=str(gpu_body.get("backend", "unknown")),
            status="online", last_seen=_now_iso(),
            gpus=[IndividualGPU(**g) for g in gpu_body.get("gpus", [])],
            serving=serving, error=None)
    except (httpx.TransportError, asyncio.TimeoutError):
        return _carry(cfg, "offline", None)
    except (ValueError, TypeError) as exc:  # bad JSON / bad shape
        return _carry(cfg, "error", f"malformed node response: {exc}")


async def poll_all_nodes_once(client: httpx.AsyncClient) -> None:
    cfgs = load_remote_nodes()
    results = await asyncio.gather(
        *(_poll_node_once(cfg, client) for cfg in cfgs),
        return_exceptions=True)
    for cfg, result in zip(cfgs, results):
        if isinstance(result, BaseException):
            logger.warning("remote node %s poll crashed: %r", cfg.name, result)
            _STATE[cfg.name] = _carry(cfg, "error", repr(result))
        else:
            _STATE[cfg.name] = result


async def poll_remote_nodes_forever(
        interval: float = POLL_INTERVAL_SECONDS) -> None:
    if not load_remote_nodes():
        logger.info("No remote nodes configured; poller idle-exits")
        return
    async with httpx.AsyncClient(timeout=NODE_TIMEOUT_SECONDS) as client:
        while True:
            try:
                await poll_all_nodes_once(client)
            except Exception:  # never die
                logger.exception("remote node poll cycle failed")
            await asyncio.sleep(interval)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ods/extensions/services/dashboard-api && python -m pytest tests/test_remote_nodes_poller.py tests/test_remote_nodes_models.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ods/extensions/services/dashboard-api/remote_nodes.py ods/extensions/services/dashboard-api/tests/test_remote_nodes_poller.py
git commit -m "feat(dashboard-api): remote node registry and poller"
```

---

### Task 7: wire poller into lifespan + merge nodes into `/api/gpu/detailed`

**Files:**
- Modify: `ods/extensions/services/dashboard-api/main.py:1032-1040` (lifespan task list)
- Modify: `ods/extensions/services/dashboard-api/routers/gpu.py:329-360` (`gpu_detailed`)
- Test: `ods/extensions/services/dashboard-api/tests/test_remote_nodes_endpoint.py`

**Interfaces:**
- Consumes: `remote_nodes.get_remote_node_statuses()`, `remote_nodes.poll_remote_nodes_forever()`.
- Produces: `/api/gpu/detailed` responses carrying `nodes`; poller running as a lifespan background task.

- [ ] **Step 1: Write the failing test**

`tests/test_remote_nodes_endpoint.py` — follow the existing pattern in `tests/` for exercising `routers/gpu.py` endpoints (see how `test_*` files there build a client or call the endpoint functions with mocks; mirror the closest existing gpu endpoint test). The test body:

```python
import pytest

import remote_nodes
from models import RemoteNodeStatus
from routers import gpu as gpu_router


@pytest.mark.asyncio
async def test_gpu_detailed_includes_remote_nodes(monkeypatch):
    gpu_router._detailed_cache["expires"] = 0.0
    gpu_router._detailed_cache["value"] = None
    monkeypatch.setattr(gpu_router, "_get_raw_gpus", lambda backend: [
        __import__("models").IndividualGPU(
            index=0, uuid="GPU-local", name="R9700", memory_used_mb=1,
            memory_total_mb=2, memory_percent=50.0, utilization_percent=3,
            temperature_c=40)])
    monkeypatch.setattr(
        remote_nodes, "get_remote_node_statuses",
        lambda: [RemoteNodeStatus(name="sparky", platform="nvidia",
                                  status="online")])
    result = await gpu_router.gpu_detailed()
    assert result.nodes[0].name == "sparky"
    assert result.gpus[0].name == "R9700"


@pytest.mark.asyncio
async def test_gpu_detailed_nodes_empty_when_unconfigured(monkeypatch):
    gpu_router._detailed_cache["expires"] = 0.0
    gpu_router._detailed_cache["value"] = None
    monkeypatch.setattr(gpu_router, "_get_raw_gpus", lambda backend: [
        __import__("models").IndividualGPU(
            index=0, uuid="GPU-local", name="R9700", memory_used_mb=1,
            memory_total_mb=2, memory_percent=50.0, utilization_percent=3,
            temperature_c=40)])
    monkeypatch.setattr(remote_nodes, "get_remote_node_statuses", lambda: [])
    result = await gpu_router.gpu_detailed()
    assert result.nodes == []
```

Adjust mock plumbing to match how `gpu_detailed` actually acquires aggregate/assignment (read the function first; monkeypatch `decode_gpu_assignment`/`_live_env_value` if they touch the filesystem).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ods/extensions/services/dashboard-api && python -m pytest tests/test_remote_nodes_endpoint.py -q`
Expected: FAIL (`nodes` empty / attribute missing)

- [ ] **Step 3: Implement**

In `routers/gpu.py`, import `import remote_nodes` at the top, and in `gpu_detailed()` add to the `MultiGPUStatus(...)` construction:

```python
        nodes=remote_nodes.get_remote_node_statuses(),
```

In `main.py` `_lifespan`, alongside the existing `asyncio.create_task(...)` entries:

```python
        asyncio.create_task(remote_nodes.poll_remote_nodes_forever()),
```

with `import remote_nodes` near the other local imports. (The poller idle-exits when unconfigured — dormant guarantee.)

- [ ] **Step 4: Run the full dashboard-api suite**

Run: `cd ods/extensions/services/dashboard-api && python -m pytest tests/ -x -q`
Expected: ALL PASS (~1187+)

- [ ] **Step 5: Commit**

```bash
git add ods/extensions/services/dashboard-api
git commit -m "feat(dashboard-api): poll remote nodes and expose them on /api/gpu/detailed"
```

---

### Task 8: UI — `RemoteNodeSection` + GPUMonitor wiring

**Files:**
- Create: `ods/extensions/services/dashboard/src/components/RemoteNodeSection.jsx`
- Modify: `ods/extensions/services/dashboard/src/pages/GPUMonitor.jsx` (append below existing content)
- Test: `ods/extensions/services/dashboard/src/components/RemoteNodeSection.test.jsx`

**Interfaces:**
- Consumes: `detailed.nodes` array from `useGPUDetailed` (shape = `RemoteNodeStatus`), existing `GPUCard` (`{ gpu }` prop, IndividualGPU shape — compatible).
- Produces: `<RemoteNodeSection node={...} />`.

- [ ] **Step 1: Write the failing test**

`RemoteNodeSection.test.jsx` (mirror the render/query idioms of an existing component test, e.g. the nearest `*.test.jsx` using @testing-library/react):

```jsx
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { RemoteNodeSection } from './RemoteNodeSection'

const gpu = {
  index: 0, uuid: 'GPU-x', name: 'NVIDIA GB10',
  memory_used_mb: 2048, memory_total_mb: 122880, memory_percent: 1.7,
  utilization_percent: 40, temperature_c: 55, power_w: 90,
  memory_type: 'unified', assigned_services: [],
}

describe('RemoteNodeSection', () => {
  it('renders online node with GPU card and serving line', () => {
    render(<RemoteNodeSection node={{
      name: 'sparky', display_name: 'DGX Spark GB10', platform: 'nvidia',
      status: 'online', last_seen: new Date().toISOString(),
      gpus: [gpu], serving: { model: 'heretic', endpoint_ok: true }, error: null,
    }} />)
    expect(screen.getByText('DGX Spark GB10')).toBeInTheDocument()
    expect(screen.getByText('NVIDIA GB10')).toBeInTheDocument()
    expect(screen.getByText(/serving/i)).toBeInTheDocument()
    expect(screen.getByText(/heretic/)).toBeInTheDocument()
  })

  it('renders offline node greyed with last seen', () => {
    const { container } = render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'offline', last_seen: new Date(Date.now() - 240000).toISOString(),
      gpus: [], serving: null, error: null,
    }} />)
    expect(screen.getByText('sparky')).toBeInTheDocument()
    expect(screen.getByText(/offline/i)).toBeInTheDocument()
    expect(screen.getByText(/last seen/i)).toBeInTheDocument()
    expect(container.firstChild.className).toMatch(/opacity-60/)
  })

  it('renders error badge distinct from offline', () => {
    render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'error', last_seen: null, gpus: [], serving: null,
      error: 'node returned HTTP 401',
    }} />)
    expect(screen.getByText(/error/i)).toBeInTheDocument()
    expect(screen.getByText(/HTTP 401/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ods/extensions/services/dashboard && npx vitest run src/components/RemoteNodeSection.test.jsx`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `RemoteNodeSection.jsx`**

```jsx
import { memo } from 'react'
import { Server } from 'lucide-react'
import { GPUCard } from './GPUCard'

const STATUS_STYLES = {
  online: { dot: 'bg-emerald-400', label: 'online', text: 'text-emerald-300' },
  offline: { dot: 'bg-zinc-500', label: 'offline', text: 'text-zinc-400' },
  error: { dot: 'bg-amber-400', label: 'error', text: 'text-amber-300' },
}

function lastSeenLabel(iso) {
  if (!iso) return null
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso)) / 60000))
  return mins < 1 ? 'last seen just now' : `last seen ${mins}m ago`
}

export const RemoteNodeSection = memo(function RemoteNodeSection({ node }) {
  const style = STATUS_STYLES[node.status] || STATUS_STYLES.offline
  const dimmed = node.status !== 'online'
  return (
    <section className={`mt-8 ${dimmed ? 'opacity-60' : ''}`}>
      <div className="flex items-center gap-3 mb-3">
        <Server size={16} className="text-zinc-400" />
        <h2 className="text-lg font-semibold text-white">
          {node.display_name || node.name}
        </h2>
        <span className="text-xs uppercase tracking-wide text-zinc-500">
          {node.platform}
        </span>
        <span className={`flex items-center gap-1.5 text-xs ${style.text}`}>
          <span className={`h-2 w-2 rounded-full ${style.dot}`} />
          {style.label}
        </span>
        {node.status !== 'online' && node.last_seen && (
          <span className="text-xs text-zinc-500">{lastSeenLabel(node.last_seen)}</span>
        )}
      </div>
      {node.error && (
        <div className="mb-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          {node.error}
        </div>
      )}
      {node.serving?.model && (
        <p className="mb-3 text-sm text-zinc-400">
          serving <span className="font-mono text-white">{node.serving.model}</span>
          {node.serving.endpoint_ok ? ' · endpoint healthy' : ' · endpoint unreachable'}
        </p>
      )}
      {node.gpus.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {node.gpus.map((gpu) => <GPUCard key={gpu.uuid} gpu={gpu} />)}
        </div>
      )}
    </section>
  )
})
```

Wire into `GPUMonitor.jsx` — import `{ RemoteNodeSection }` and append immediately before the component's closing wrapper element:

```jsx
      {(detailed?.nodes ?? []).map((node) => (
        <RemoteNodeSection key={node.name} node={node} />
      ))}
```

- [ ] **Step 4: Run dashboard tests**

Run: `cd ods/extensions/services/dashboard && npx vitest run`
Expected: ALL PASS (124 existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add ods/extensions/services/dashboard/src
git commit -m "feat(dashboard): render remote node sections on GPU page"
```

---

### Task 9: deploy to metal + smoke battery

**Files:** none in-repo (runtime deployment; document results in `~/notes/reports/`)

**Interfaces:**
- Consumes: everything above; autarch runtime at `~/ods`; sparky via `ssh sparky`.

- [ ] **Step 1: Generate the node key and deploy node-agent to sparky**

```bash
KEY=$(openssl rand -hex 24)
rsync -a --delete ~/projects/ODS/ods/extensions/services/node-agent ~/projects/ODS/ods/extensions/services/dashboard-api sparky:ods-src/ods/extensions/services/
ssh sparky "cd ods-src && NODE_AGENT_KEY=$KEY NODE_NAME=sparky NODE_SERVING_PROBE_URL=http://localhost:8000/v1/models NODE_SERVING_CONTAINER=aeon-vllm docker compose -f ods/extensions/services/node-agent/compose.yaml up -d --build"
```

(rsync only the two service dirs — the Dockerfile references nothing else; create parent dirs with `ssh sparky 'mkdir -p ods-src/ods/extensions/services'` first.)

- [ ] **Step 2: Firewall on sparky (Tim's hands — sudo)**

Stage and hand to Tim in a real terminal:

```bash
ssh -t sparky 'sudo ufw allow from 192.168.1.6 to any port 7720 proto tcp'
```

- [ ] **Step 3: Verify agent from autarch**

```bash
curl -s -H "Authorization: Bearer $KEY" http://192.168.1.15:7720/v1/node/gpu | python3 -m json.tool | head -20
curl -s -o /dev/null -w "%{http_code}\n" http://192.168.1.15:7720/v1/node/gpu   # expect 401
```

Expected: real GB10 metrics; unauthenticated 401.

- [ ] **Step 4: Configure autarch dashboard-api + deploy fork build**

Add to `~/ods/.env`:

```
ODS_REMOTE_NODES=[{"name":"sparky","display_name":"DGX Spark GB10","url":"http://192.168.1.15:7720","key_env":"ODS_NODE_KEY_SPARKY"}]
ODS_NODE_KEY_SPARKY=<the generated key>
```

Then sync the three changed services from `~/projects/ODS` into the runtime tree and rebuild, following the exact deploy pattern used for the Model Deck work (check `~/notes/ods-delivery-manifest.md` for the canonical sync command; ensure the compose service env passes `ODS_REMOTE_NODES` and `ODS_NODE_KEY_SPARKY` through to dashboard-api — add to the dashboard-api service `environment:` block if not using `env_file`).

```bash
cd ~/ods && docker compose up -d --build dashboard-api dashboard
```

- [ ] **Step 5: Metal smoke battery**

```bash
# nodes present in payload
curl -s -H "Authorization: Bearer $(docker exec ods-dashboard-api sh -c 'echo $ODS_API_KEY' 2>/dev/null || echo unknown)" http://127.0.0.1:3002/api/gpu/detailed | python3 -c "import json,sys; d=json.load(sys.stdin); print([n['name'] for n in d.get('nodes',[])])"
```

(Confirm the dashboard-api auth mechanism from `nginx.conf`/existing smoke scripts if the key lookup differs.) Then in the browser: GPU page shows the sparky section live. Then resilience:

```bash
ssh sparky 'docker stop ods-node-agent'   # section goes grey within ~10s, autarch cards unaffected
ssh sparky 'docker start ods-node-agent'  # section returns online
```

- [ ] **Step 6: Record + push**

Update `~/notes/reports/<today>.md` and memory; push the branch:

```bash
cd ~/projects/ODS && git push fork feat/remote-node-gpu-visibility
```

---

## Self-review notes (completed)

- **Spec coverage:** protocol (T1–3), single-source collectors + deploy (T4), additive payload + contract (T5), registry/poller/states/isolation (T6), dormancy + wiring (T7), grouped UI + 3 states (T8), metal smoke incl. kill-agent (T9). Capabilities seam: `/v1/node/info` returns `capabilities` (T1); actions namespace deliberately absent.
- **Type consistency:** `RemoteNodeStatus`/`RemoteNodeServing` used identically in T5–T8; node-agent responses match what the poller parses (`backend`/`gpus`, serving triple).
- **Known judgment calls for the implementer:** async-test marker style must match the suite's existing plugin (T6/T7 note); T7 mock plumbing must be adapted to `gpu_detailed()`'s real dependencies after reading it; T9 auth lookup for the smoke curl must be confirmed against the runtime.
