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
| `NODE_SERVING_PROBE_URL` | *(unset)* | OpenAI-compatible `/v1/models`-style URL to probe for what model is currently being served. Left unset disables the probe. This is the default for vllm-engine profiles; overridden per-profile by profiles.json (a non-vLLM profile's `health_url` takes over instead — see [`GET /v1/node/profiles`](#get-v1nodeprofiles)). |
| `NODE_SERVING_CONTAINER` | *(unset)* | Name of the local Docker container running inference; its status is reported via `docker inspect`. **Requires opting into the Docker socket mount, which grants host-root-equivalent access — see [Security](#security).** Left unset (recommended) disables the check and `container_status` is reported as `null`. Also the fallback when the current profile's `container` field in profiles.json is unset. |
| `NODE_AGENT_PORT` | `7720` | Port uvicorn binds inside the container (also the host port under `network_mode: host`). Read by the Dockerfile `CMD`, not by the Python config. |
| `NODE_AGENT_BIND` | `0.0.0.0` | Address uvicorn binds. The default is every interface on the node — including any WAN or management NIC. On a multi-homed node, set this to the one address the dashboard host reaches it on. The healthcheck follows it. Read by the Dockerfile `CMD`, not by the Python config. |
| `NODE_GPU_CACHE_TTL` | `2.0` | TTL in seconds for the short-lived GPU sample cache used by `/v1/node/gpu`. |
| `NODE_VLLM_DIR` | *(unset)* | Read-only mount of the vLLM profile directory (`compose-<profile>.yaml` set, plus an optional `profiles.json` metadata sidecar). Both this and `NODE_SWAP_CTL_DIR` must be set to enable `/v1/node/profiles` and `/v1/node/swap`; either unset and both answer `503`. |
| `NODE_SWAP_CTL_DIR` | *(unset)* | Shared file-protocol directory this agent writes `request.json` into and reads the host-side swap-helper's `status.json` from. See [`POST /v1/node/swap`](#post-v1nodeswap). |

## Deploy

Runs on the **remote** node, built from a full checkout of this repo (the
build context is the repo root so it can vendor `dashboard-api/gpu.py`;
`Dockerfile.dockerignore` keeps that context from shipping the whole repo):

```bash
docker compose -f ods/extensions/services/node-agent/compose.yaml.disabled up -d --build
```

The compose fragment ships as `compose.yaml.disabled` on purpose. This service
is **not** part of the local ODS stack, and `ods` builds its compose `-f` list
from every extension whose `compose.yaml` exists — the `.disabled` suffix is
the repo's existing convention for keeping a fragment out of that set. Do not
run `ods enable node-agent`; deploy it by hand on the remote host as above.

The compose file requests the `nvidia` device driver with the `utility`
capability, which is what makes `nvidia-smi` available inside the
otherwise-slim container. On non-NVIDIA nodes, drop that `deploy.resources`
block and set `GPU_BACKEND` accordingly.

**Surviving a reboot of the remote node** is Docker's job here: the fragment
sets `restart: unless-stopped`, so the agent comes back on its own — but only
if the Docker daemon itself starts at boot. That is not automatic on every
distribution, so confirm it on the remote host:

```bash
systemctl is-enabled docker   # expect "enabled"; otherwise: sudo systemctl enable docker
```

There is no systemd unit to install for the agent itself.

### Registering the node with the dashboard host

On the **dashboard** host, add the node to `ODS_REMOTE_NODES` and its key to
`ODS_REMOTE_NODE_KEYS` (see `ods/.env.example`), then **recreate the
dashboard-api container** — the remote-node config is read once at startup, so
a running dashboard-api will not pick up a new node.

For a key that should not sit in the container's environment, put the same
`{"<node name>": "<bearer key>"}` object in a 0600 file, mount it into the
dashboard-api container, and point `ODS_REMOTE_NODE_KEYS_FILE` at the path
*inside* that container. It wins per node over `ODS_REMOTE_NODE_KEYS`, and
because it is re-read each poll cycle, rotating this node's key afterwards is
a file write on the dashboard host plus a restart of the agent here — no
dashboard-api recreate. Write the file atomically (write a temp file, then
`mv` over it): a half-written read falls back to the env map for that cycle.

## Security

- **The Docker socket mount grants host-root-equivalent access.** Mounting
  `/var/run/docker.sock` lets this container drive the full Docker API: create
  privileged containers, bind-mount the host filesystem, read every secret.
  Adding `:ro` write-protects only the socket *file* — the API behind it stays
  fully writable — so `:ro` **does not mitigate this at all**. Because this
  service is network-exposed, anyone who compromises it inherits host root.
- Both the socket and the `docker` CLI mounts are therefore **commented out by
  default**. Leave `NODE_SERVING_CONTAINER` unset and the mounts commented
  unless you actively accept that trade; without them the agent works
  unchanged and simply reports `container_status: null`.
- **Always firewall-scope port 7720** to the dashboard host (see [Ports](#ports)).
  The service uses `network_mode: host`, so by default it binds every interface
  on the node; `NODE_AGENT_BIND` narrows that to one address.
- Every route is bearer-gated with a constant-time comparison, and the
  unauthenticated OpenAPI surface (`/docs`, `/redoc`, `/openapi.json`) is
  disabled so the API is not advertised to whoever can reach the port.
- The agent is read-only for metrics and probes, with one deliberate exception:
  `POST /v1/node/swap` asks the host swap-helper to switch between
  operator-approved compose profiles. It still exposes no way to load, unload,
  start, or stop arbitrary workloads — the helper revalidates every request and
  only accepts profiles whose compose files the operator placed on the node.

## Ports

| Port | Protocol | Where | Binds | Purpose |
|---|---|---|---|---|
| `NODE_AGENT_PORT` (7720) | TCP | Remote node | every interface by default under `network_mode: host` — narrow with `NODE_AGENT_BIND` | Bearer-gated read-only GPU/serving metrics, polled by the dashboard host |

The agent opens nothing else: no second listener, no discovery broadcast, no
port on the dashboard host. Scope inbound access to the dashboard host only:

```bash
ufw allow from <dashboard-ip> to any port 7720 proto tcp
```

On a multi-homed node, `NODE_AGENT_BIND=<lan-ip>` narrows the listener to one
interface as well. Treat that as defence in depth, not a replacement for the
firewall rule — it still accepts every host that can reach that address.

## API

All endpoints require `Authorization: Bearer <NODE_AGENT_KEY>` and return
`401` (empty body) if the header is missing or wrong.

### `GET /v1/node/info`

Node identity plus a fresh (uncached) GPU inventory.

> Currently **unconsumed** by the dashboard: the poller only calls
> `/v1/node/gpu` and `/v1/node/serving`. `/v1/node/info` is the phase-2
> capabilities seam (`capabilities: ["metrics"]`), kept so a later phase can
> negotiate what a node supports without a protocol change.

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
  "gpus": [{"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GB10", "memory_used_mb": 1024, "memory_total_mb": 122880, "memory_percent": 0.8, "utilization_percent": 5, "temperature_c": 45, "power_w": 30.0, "memory_type": "unified"}],
  "error": null
}
```

`error` is nullable and reports a *collector* failure while the node itself is
perfectly reachable: the collector is absent, or it ran and produced nothing
usable. That is distinct from a node which genuinely has no GPUs, and which
answers `"gpus": [], "error": null`. The dashboard keeps such a node `online`
and displays the message on its card. The field is additive — a dashboard-api
predating it simply ignores it.

```json
{
  "backend": "nvidia",
  "gpus": [],
  "error": "GPU collector unavailable: no usable 'nvidia' collector on this node (check that the vendor SMI tool is installed and that the GPU devices are exposed to this container)"
}
```

### `GET /v1/node/serving`

Probes what this node is serving. The probe is **engine-aware**: it first
asks swap control (see below) which profile is current and what engine it
runs.

- **Swap control disabled, or no current profile** (the plain env-configured
  case, and the only one that existed before profiles.json): probes the
  OpenAI-compatible `/v1/models` endpoint at `NODE_SERVING_PROBE_URL` and/or
  the local Docker container named by `NODE_SERVING_CONTAINER`, exactly as
  described in the env var table above.
- **Current profile's engine is `vllm`** (including profiles.json's default
  when a profile has no explicit `engine`): identical to the above — the
  env-configured OpenAI probe path, unchanged.
- **Current profile's engine is anything else** (e.g. `comfyui`): `model` is
  the profile's name (no `/v1/models` call, no OpenAI-shaped parse);
  `endpoint_ok` is `true` for any 2xx response from the profile's
  `health_url` in profiles.json (unset means unprobed, `endpoint_ok: false`);
  `container_status` uses the profile's `container` from profiles.json,
  falling back to `NODE_SERVING_CONTAINER` when the profile doesn't set one.

```bash
curl -H "Authorization: Bearer $NODE_AGENT_KEY" http://<node-ip>:7720/v1/node/serving
```

```json
{"model": "heretic", "endpoint_ok": true, "container_status": "running"}
```

Same shape for a non-vLLM profile, e.g. a `comfyui` swap:

```json
{"model": "comfyui", "endpoint_ok": true, "container_status": "running"}
```

### `GET /v1/node/profiles`

Lists the available vLLM-directory profiles (the `compose-*.yaml` set under
`NODE_VLLM_DIR`) and the host-side swap-helper's last-known status. Answers
`503` unless both `NODE_VLLM_DIR` and `NODE_SWAP_CTL_DIR` are configured.

```bash
curl -H "Authorization: Bearer $NODE_AGENT_KEY" http://<node-ip>:7720/v1/node/profiles
```

Each entry in `profiles` is a metadata **object**, not a bare name — sourced
from an optional `<NODE_VLLM_DIR>/profiles.json` sidecar
(`{"<profile>": {"engine": ..., "health_url": ..., "container": ...}}`),
defaulting to `{"engine": "vllm", "health_url": null, "container": null}` for
any profile missing from that file, or when the file itself is absent or
unreadable:

```json
{
  "profiles": [
    {"name": "comfyui", "engine": "comfyui", "health_url": "http://127.0.0.1:8188/system_stats", "container": "spark-comfyui"},
    {"name": "heretic", "engine": "vllm", "health_url": null, "container": null}
  ],
  "swap_status": {"state": "done", "profile": "heretic", "id": "abc123", "message": "swap launched", "ts": "2026-07-30T22:00:00Z"}
}
```

`swap_status` is `null` until the swap-helper has written its first
`status.json` (i.e. before any swap has ever been requested on this node).
`engine`/`health_url`/`container` are what `/v1/node/serving` consults once
that profile becomes current — see above.

### `POST /v1/node/swap`

Requests a profile swap. The agent never touches Docker itself: this only
writes `request.json` into `NODE_SWAP_CTL_DIR` for the host-side
swap-helper (`swap-helper/swap-helper.sh`) to pick up, validate again, and
run `swap.sh`. Poll `/v1/node/profiles`' `swap_status` for progress. Answers
`503` unless both `NODE_VLLM_DIR` and `NODE_SWAP_CTL_DIR` are configured.

```bash
curl -X POST -H "Authorization: Bearer $NODE_AGENT_KEY" \
     -H "Content-Type: application/json" \
     -d '{"profile": "heretic"}' \
     http://<node-ip>:7720/v1/node/swap
```

```json
{"id": "3f9c9b2e-...", "profile": "heretic"}
```

| Status | Meaning |
|---|---|
| `202` | Request accepted and written; swap is now the helper's job. |
| `400` | Profile name fails the `^[A-Za-z0-9_-]+$` check. |
| `404` | No `compose-<profile>.yaml` for that name under `NODE_VLLM_DIR`. |
| `409` | A request is already pending, or the helper is mid-swap. |
