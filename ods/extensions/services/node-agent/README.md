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
| `NODE_SETTINGS_DIR` | *(unset)* | Shared file-protocol directory holding per-profile settings documents (`<profile>.json`, written by this agent's settings routes) and harvested catalog files (`catalog-<profile>.json`, written by the host-side swap-helper). Unset disables `/v1/node/profile/{profile}/settings` and `/v1/node/catalog`, which answer `503`. |
| `NODE_INSTANCES_CTL_DIR` | *(unset)* | Shared file-protocol directory this agent writes `instance-req.json` into and reads the host-side instances-helper's `instance-status-<resource>.json` from (INST I1). Unset disables `/v1/node/instance/{resource}` and its status route, which answer `503` (`instances.InstancesDisabled`), and drops `"instances"` from `/v1/node/info`'s `capabilities`. See [Engine instances](#engine-instances-inst-i1) below. |

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
launch. Poll `/v1/node/profiles`' `swap_status` for progress. Answers
`503` unless both `NODE_VLLM_DIR` and `NODE_SWAP_CTL_DIR` are configured.

How the helper launches depends on its OPTIONAL 4th argument
(`swap-helper.sh --daemon <ctl-dir> <vllm-dir> [<settings-dir>]`, the same
directory as `NODE_SETTINGS_DIR`):

* **Without it** — and whenever `<settings-dir>/<profile>.json` is missing
  or unusable — the helper shells out to `<vllm-dir>/swap.sh` exactly as it
  always has, and deletes any stale override it had left behind. A settings
  fault can therefore never break a swap.
* **With a valid document**, the helper renders it to
  `<vllm-dir>/settings-<profile>.override.yaml` (`command` and
  `environment` only; never a `compose-*.yaml` name, which
  `/v1/node/profiles` would then list as a ghost profile), tears down every
  container named across `compose-*.yaml`, and runs `docker compose -f
  compose-<profile>.yaml -f settings-<profile>.override.yaml up -d`.
  Compose replaces `command` outright and merges `environment` over the base
  file's, so image/volumes/devices stay the node operator's.

After a successful launch of a profile whose `profiles.json` engine is
`vllm` (the default), the helper runs `swap-helper/harvest_probe.py` inside
the new container over `docker exec -i` and writes
`<settings-dir>/catalog-<profile>.json` —
`{image_id, harvested_ts, engine, probe_output}`, served by
`GET /v1/node/catalog`. The probe is best-effort: any failure is logged and
the swap outcome stands. `harvest_probe.py` is generated from
`model-deck`'s `app.harvest.PROBE_SOURCE` and pinned byte-identical to it by
a test — edit the constant, not this copy.

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

### Engine instances (INST I1)

A second, independent file-protocol channel: the Model Deck's own
`POST /api/nodes/{id}/instances` (create/remove/move — see model-deck's
`README.md`'s **Engine instances** section for the operator-facing shape)
reaches this node through here. Same posture as swap control above — this
agent never touches Docker itself, only validates the document's SHAPE and
queues a file for a host-side helper — but it is a **separate** ctl
directory, a separate helper, and a separate `docker compose` project
(`deck-instances`), never the `NODE_VLLM_DIR`/swap-helper machinery above.
Kind names are deliberately **not known here** (`instances.py`'s module
docstring): the agent validates `resource`/`gpu_indices`/`port`/`env`
shape only, and it is the host-side instances-helper that resolves `kind`
against its own `templates/kinds.json` — so a compromised agent can at
most ask the helper for one of the *operator's own* templates, never an
arbitrary image or command.

#### `POST /v1/node/instance/{resource}`

Body: `{"verb": "create"|"remove"|"move", "document": {...}}`. The
document is the wire shape the deck's `app/instances.py::instance_document`
builds — exactly `{resource, kind, gpu_indices, port, env}`, no more, no
fewer keys (`instances.py`'s `DOC_KEYS`) — and `document["resource"]` must
equal the path `{resource}`. Writes `<NODE_INSTANCES_CTL_DIR>/instance-req.json`
for the instances-helper to pick up; nothing here reads a result back for
control flow — the deck observes the container itself, never this file.

```bash
curl -X POST -H "Authorization: Bearer $NODE_AGENT_KEY" \
     -H "Content-Type: application/json" \
     -d '{"verb": "create", "document": {"resource": "lemonade-1", "kind": "lemonade", "gpu_indices": [2], "port": 18100, "env": {}}}' \
     http://<node-ip>:7720/v1/node/instance/lemonade-1
```

```json
{"accepted": true}
```

| Status | Meaning |
|---|---|
| `202` | Request accepted and written; the instances-helper now owns it. |
| `409` | A request is already pending for this node (one in-flight request at a time — `instance-req.json` already exists). This is exactly the 409 the deck itself turns into `app.engines.BusyError` (`app/engines/instances.py`) and re-raises as its OWN 409 on all three of `POST/DELETE/POST .../instances[/move]` (`app/routers/instances.py`'s `_ship`) — an operator seeing a 409 from the deck's instances routes should look here, not at a declaration conflict. |
| `422` | The document fails shape validation (wrong key set, bad `resource`/`gpu_indices`/`port`/`env`), or `document["resource"] != {resource}`. |
| `503` | `NODE_INSTANCES_CTL_DIR` is unset — instance control is not enabled on this node. |

#### `GET /v1/node/instance/{resource}/status`

Reads back `<NODE_INSTANCES_CTL_DIR>/instance-status-<resource>.json`,
the instances-helper's own completion record (`{resource, verb, ok, error,
ts}`) — **forensics for a human**, never consulted by the deck's own
observation, which watches the container directly. `null` result means no
status file exists yet (never requested, or a stale one already cleared
before the slow part of the verb began).

```bash
curl -H "Authorization: Bearer $NODE_AGENT_KEY" http://<node-ip>:7720/v1/node/instance/lemonade-1/status
```

```json
{"result": {"resource": "lemonade-1", "verb": "create", "ok": true, "error": null, "ts": "2026-08-20T12:00:00Z"}}
```

### Deploy: the instances overlay + host-side helper

Instance control is **opt-in on top of** the base agent deploy above, in
two pieces that both have to be running:

**1. The agent's own overlay** (`compose.instances.yaml.disabled`) adds
`NODE_INSTANCES_CTL_DIR` and mounts the shared ctl directory into the
agent container:

```bash
docker compose -f compose.yaml.disabled -f compose.instances.yaml.disabled up -d --build
```

Point `HOST_INSTANCES_CTL_DIR` at wherever the ctl directory should live
on the host (e.g. `~/deck-instances/ctl`) — the overlay's own
`:?path to the instances ctl dir` guard fails fast if it's unset. The base
compose file's NVIDIA `deploy.resources` block is reset to nothing here
(`deploy: !reset {}`): the local node's own GPU observation for instances
stays on the deck's sysfs reader, so the agent needs no device access at
all for this channel — the block only ever applied to the sparky-style
remote deployment above.

**2. The host-side instances-helper** (`instances-helper/`) — the
privileged half that actually runs `docker compose` on the rendered
per-kind template. This is Python + one bash script; it does not run in a
container, and it must run from a **full checkout of this repo**, not a
copy — the deployed `~/ods/extensions/services/node-agent/` tree is a
build artifact snapshot and goes stale the moment this repo's templates or
scripts change; point the systemd unit (below) at the repo checkout, never
at `~/ods/...`.

```bash
instances-helper/instances-helper.sh --daemon \
    <ctl-dir> instances-helper/templates <instances-dir> <ods-dir>
```

- `<ctl-dir>` — same directory as `HOST_INSTANCES_CTL_DIR` above (this is
  the two halves of the same file-protocol pair).
- `instances-helper/templates` — `kinds.json` (kind → template filename)
  plus one JSON template per kind (`hipfire.json`, `lemonade.json`,
  `comfyui.json`) declaring the image, internal port, compose service
  block, environment defaults, `env_allow` (the ONLY env keys a deck
  document may override — see model-deck's `README.md`), the per-instance
  data directories to create, and (for kinds that serve a fixed model) a
  `route` block consumed by `stage_route.py`.
- `<instances-dir>` — where the helper renders `<resource>.yaml` (one
  compose file per instance) and creates `<instances-dir>/data/<resource>/`
  with that kind's `per_instance_dirs` underneath.
- `<ods-dir>` — the operator's ODS checkout, read-only from here: template
  volumes reference it (e.g. lemonade's model directory), and
  `stage_route.py` writes into its `config/litellm/extra-routes.json` —
  **staging** an instance's gateway route only (D-I1-4). *Applying* a
  staged route — regenerating LiteLLM's config and recreating the
  container — is the existing ODS activate/regen path; this helper never
  triggers it.

Every render+launch runs under docker compose project `deck-instances` —
never the ODS stack's own multi-file project, and never with
`--remove-orphans` (Global Constraints: mixing projects that way can tear
down containers the OTHER project owns).

**Supervision (D-I1-7):** a systemd **user** unit
(`instances-helper/deck-instances-helper.service`), not a system unit or
cron — the operator runs `docker compose` as themself, and a user unit
keeps that identity:

```bash
systemctl --user daemon-reload
systemctl --user enable --now deck-instances-helper.service
loginctl enable-linger "$USER"   # keeps the unit running after logout —
                                  # without this, the unit dies at the next
                                  # SSH disconnect
```

The unit's `ExecStart` in this repo is templated on `%h` (the invoking
user's home) pointing at a full checkout under `~/projects/ODS/...` — edit
it to match wherever this repo is actually checked out before enabling it;
the checkout-not-copy rule above applies to whatever path it names.

### Security note: the local-instances address

When `control: "instances"` is set on the Model Deck's **local** node
entry (the common case: instances running beside the deck on the same
box), the deck reaches this agent over `ods-network`'s gateway address,
not `localhost` and not `host.docker.internal` (which does not reach a
`network_mode: host` process — see `app/settings.py`'s comment on this).
**The deck's `address` for the local node is `http://172.18.0.1:7720`.**
Firewall that address class, not a single host IP:

```bash
ufw allow from 172.18.0.0/16 to any port 7720 proto tcp
```

`172.18.0.0/16` is `ods-network`'s bridge subnet — every container on it,
including the deck, presents as some address in that range, and the
gateway address itself can shift if the network is ever recreated. This is
the same class of mistake as the 2026-07-16 host-agent UFW incident
(`ods-ufw-blocks-hostagent-on-reboot`): scoping to one observed IP instead
of the subnet silently relocks the agent out from under the deck the next
time the bridge is rebuilt.
