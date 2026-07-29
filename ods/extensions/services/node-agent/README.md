# ods-node-agent

Read-only metrics sidecar for **remote** inference nodes (e.g. a DGX-class
box or any second GPU host) that don't run the full ODS stack. It exposes a
small, bearer-authenticated HTTP API that the ODS dashboard-api polls to
show that node's GPUs and what it's serving, giving multi-node GPU
visibility without deploying dashboard-api itself on the remote box.

GPU collection logic (`gpu.py`) is vendored at build time from
`dashboard-api` (single source of truth) — see the Dockerfile. The response
models in `models.py` are a local shim kept in field-parity with
`dashboard-api/models.py`'s `GPUInfo`/`IndividualGPU`; `tests/test_model_parity.py`
fails the build if they drift apart (that test only runs inside a full repo
checkout — it self-skips in the deployed container, where `dashboard-api/`
isn't present).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NODE_AGENT_KEY` | *(required, no default)* | Bearer token clients must send as `Authorization: Bearer <key>`. Compose fails fast (`:?set NODE_AGENT_KEY`) if unset. |
| `NODE_NAME` | hostname | Human-readable node identifier returned by `/v1/node/info`. |
| `GPU_BACKEND` | `nvidia` | Which GPU backend to collect from (`nvidia`, `amd`, `apple`). |
| `NODE_SERVING_PROBE_URL` | *(unset)* | OpenAI-compatible `/v1/models`-style URL to probe for what model is currently being served. Left unset disables the probe. |
| `NODE_SERVING_CONTAINER` | *(unset)* | Name of the local Docker container running inference; its status is reported via `docker inspect`. Left unset disables the check. |
| `NODE_AGENT_PORT` | `7720` | Port uvicorn binds inside the container (also the host port under `network_mode: host`). |

Advanced/optional (not templated in `compose.yaml`, set via the host
environment if needed): `NODE_GPU_CACHE_TTL` (seconds, default `2.0`) —
TTL for the short-lived GPU sample cache used by `/v1/node/gpu`.

## Deploy

Runs on the **remote** node, built from a full checkout of this repo (the
build context is the repo root so it can vendor `dashboard-api/gpu.py`):

```bash
docker compose -f ods/extensions/services/node-agent/compose.yaml up -d --build
```

The compose file requests the `nvidia` device driver with the `utility`
capability, which is what makes `nvidia-smi` available inside the
otherwise-slim container. On non-NVIDIA nodes, drop that `deploy.resources`
block and set `GPU_BACKEND` accordingly.

## Firewall

This service listens on `network_mode: host`, so scope inbound access to
the dashboard host only, e.g.:

```bash
ufw allow from <dashboard-ip> to any port 7720 proto tcp
```

## API

All endpoints require `Authorization: Bearer <NODE_AGENT_KEY>` and return
`401` (empty body) if the header is missing or wrong.

### `GET /v1/node/info`

Node identity plus a fresh (uncached) GPU inventory.

```bash
curl -H "Authorization: Bearer $NODE_AGENT_KEY" http://<node-ip>:7720/v1/node/info
```

```json
{
  "name": "aeon",
  "hostname": "aeon",
  "platform": "nvidia",
  "capabilities": ["metrics"],
  "gpus": [{"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GB10", "memory_used_mb": 1024, "memory_total_mb": 122880, "memory_percent": 0.8, "utilization_percent": 5, "temperature_c": 45, "power_w": 30.0, "memory_type": "unified"}]
}
```

### `GET /v1/node/gpu`

GPU inventory only, served from a short-lived cache (`NODE_GPU_CACHE_TTL`
seconds) to keep polling cheap.

```bash
curl -H "Authorization: Bearer $NODE_AGENT_KEY" http://<node-ip>:7720/v1/node/gpu
```

```json
{
  "backend": "nvidia",
  "gpus": [{"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GB10", "memory_used_mb": 1024, "memory_total_mb": 122880, "memory_percent": 0.8, "utilization_percent": 5, "temperature_c": 45, "power_w": 30.0, "memory_type": "unified"}]
}
```

### `GET /v1/node/serving`

Probes what this node is serving: an OpenAI-compatible `/v1/models`
endpoint (`NODE_SERVING_PROBE_URL`) and/or a local Docker container's
status (`NODE_SERVING_CONTAINER`).

```bash
curl -H "Authorization: Bearer $NODE_AGENT_KEY" http://<node-ip>:7720/v1/node/serving
```

```json
{"model": "heretic", "endpoint_ok": true, "container_status": "running"}
```
