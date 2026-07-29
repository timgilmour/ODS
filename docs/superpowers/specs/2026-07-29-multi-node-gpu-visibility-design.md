# Multi-Node GPU Visibility — Design

**Date:** 2026-07-29
**Status:** Approved by Tim (brainstorm session, autarch)
**Branch:** `feat/remote-node-gpu-visibility` (cut from upstream-synced `main`)

## Problem

ODS's GPU page shows only the GPUs of the machine running the stack. The dashboard-api
collects metrics locally (`gpu.py`) and talks to exactly one host-agent (`AGENT_URL`).
Homelabs increasingly have additional inference boxes (here: a DGX Spark "sparky",
GB10, serving vLLM behind litellm) that are invisible to the dashboard.

Goal: remote inference nodes appear on the GPU page using core ODS machinery —
generic, upstreamable, not Sparky-specific.

## Decisions (locked with Tim)

1. **Scope now: read-only monitoring.** Remote control (model swap, restarts) is a
   later phase; this design must leave a clean seam for it, not build it.
2. **Node side: slim node-agent** (new small service), not the full host-agent, not
   agentless SSH.
3. **UI: grouped by host.** Existing local layout untouched; remote nodes get their
   own sections. Remote GPUs never appear in deck topology/assignments.
4. **Upstream quality:** generic "remote nodes" feature, TDD, contract tests,
   config-driven, dormant when unconfigured.

## Architecture

Three parts: a node-agent on each remote box, a registry+poller in dashboard-api,
and host-grouped rendering in the dashboard UI. The browser only ever talks to
dashboard-api.

```
[node-agent @ sparky:7720]  <-- poll every ~5s --  [dashboard-api poller]
  nvidia-smi collector                                 registry: ODS_REMOTE_NODES
  serving probe (vllm)                                 cache + staleness
                                                          |
                                                   /gpu detailed payload
                                                   (+ additive `nodes` array)
                                                          |
                                                   [GPUMonitor.jsx]
                                                   autarch section (unchanged)
                                                   + one section per node
```

### Node protocol (versioned, `/v1/`)

- `GET /v1/node/info` — `{name, hostname, platform, gpus: [inventory], capabilities}`.
  `capabilities` is `["metrics"]` today. Phase-2 agents will advertise e.g.
  `"actions:model-swap"`; dashboard gates UI on capabilities. This is the
  remote-control seam.
- `GET /v1/node/gpu` — live metrics in the dashboard-api `GPUInfo` shape, produced by
  the same collector code dashboard-api uses (single-sourced at container build time
  from `dashboard-api/gpu.py`; no fork/copy of collector logic).
- `GET /v1/node/serving` — `{model, endpoint_ok, container_status}` from probing a
  configurable local OpenAI endpoint (default `http://localhost:8000/v1/models`) and
  optionally a named container.
- Auth: static bearer key per node (`NODE_AGENT_KEY`), same pattern as
  `ODS_AGENT_KEY`. Unauthenticated requests: 401, no body detail.
- Reserved, unbuilt: `/v1/node/actions/*`.

### node-agent service

- Location: `ods/extensions/services/node-agent/` — FastAPI, same stack as
  dashboard-api. Dockerfile builds from repo root so it can vendor
  `dashboard-api/gpu.py` + `models.py` as a module at build time (single source).
- Deployment: single container via a small compose file on the remote node; env:
  `NODE_AGENT_KEY`, `NODE_AGENT_PORT` (default 7720; host-agent owns 7710), `NODE_SERVING_PROBE_URL`,
  `NODE_SERVING_CONTAINER` (optional). No ODS stack required on the node.
- Local metrics TTL cache (~2s) so dashboard polling can't spam nvidia-smi.
- Network posture on the node: bind all interfaces, rely on host firewall scoping
  (documented); Sparky's ufw allows the port from autarch only.

### dashboard-api changes

- Config: `ODS_REMOTE_NODES` — JSON array `[{"name": "sparky", "display_name":
  "DGX Spark GB10", "url": "http://192.168.1.15:7720", "key_env":
  "ODS_NODE_KEY_SPARKY"}]`. Keys are referenced by env-var name, never inline.
  Absent/empty config → feature fully dormant, zero behavior change.
- Background poller task (pattern: existing services poll): every ~5s per node,
  2s timeout, per-node isolation — one bad node never delays local data or other
  nodes. Results cached with `last_seen`.
- Detailed-GPU payload gains an **additive** `nodes` array:
  `{name, display_name, platform, status: online|offline|error, last_seen,
  gpus: [...], serving: {...}, error: str|null}`. Every existing field is
  byte-for-byte unchanged — enforced by contract test.
- Node states: **online** (fresh data), **offline** (connect/timeout),
  **error** (401/malformed — distinct from offline so a bad key doesn't read as
  "box down"). Node up but collector failing → `status: online, gpus: [],
  error: "<collector message>"`.

### Dashboard UI changes

- `GPUMonitor.jsx`: existing local rendering (cards, charts, topology, deck
  assignments) pixel-identical. Below it, one section per configured node:
  header `sparky · DGX Spark GB10 · ● online`, reusing `GPUCard` for metrics,
  plus a serving line ("serving heretic · endpoint healthy").
- Offline: section greyed, "last seen Nm ago". Error: distinct badge + message.
- No remote entries in AssignmentTable/TopologyView — remote nodes are not deck
  resources.

## Testing (TDD, #1783 discipline)

- **dashboard-api pytest:** registry parsing (incl. malformed config), poller merge,
  state transitions (online→offline→online, auth error), per-node isolation, and the
  **back-compat contract test**: detailed payload schema unchanged with empty AND
  populated node lists.
- **node-agent pytest:** auth (401 without/with wrong key), collector adapter,
  serving probe against mocked endpoint/container states, TTL cache behavior.
- **dashboard vitest:** node sections render; offline/error states; existing
  GPUMonitor tests unmodified and green.
- **Metal smoke:** curl agent endpoints on sparky; dashboard shows the section live;
  kill agent → section goes grey without disturbing autarch data.

## Out of scope (explicit)

- Remote actions of any kind (phase 2, lands on `/v1/node/actions/*` + capabilities).
- Deck/placement integration for remote GPUs.
- Auto-discovery (mDNS etc.) — nodes are statically configured.
- AMD/Apple remote nodes are *supported by construction* (collector module covers
  them) but only NVIDIA is validated on metal in this phase.

## Rollout

1. Build + tests on `feat/remote-node-gpu-visibility`.
2. Deploy to autarch's runtime (`~/ods`) + node-agent container on sparky; metal smoke.
3. Run for a few days; then decide on the upstream PR (this spec's docs commit gets
   dropped or kept per PR-crafting preference at that time).
