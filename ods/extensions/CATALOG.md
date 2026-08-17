# ODS Extension Catalog

This catalog lists all **bundled extensions** (services) that ship with ODS. Each extension has a `manifest.yaml` that declares its id, ports, health endpoint, and **version compatibility** (`compatibility.ods_min` / `ods_max`) so they work seamlessly for the ODS version you are on.

For adding or authoring extensions, see [EXTENSIONS.md](../docs/EXTENSIONS.md) and [schema/README.md](schema/README.md).

## Catalog overview

| Service ID      | Name                     | Category    | Default port | GPU backends   | Description |
|-----------------|--------------------------|------------|-------------|----------------|-------------|
| llama-server    | llama-server (LLM)       | core       | 8080        | amd, nvidia    | Main OpenAI-compatible LLM inference API. Linux Docker host exposure defaults to `OLLAMA_PORT=11434`; native macOS/Windows paths use host `8080`. |
| open-webui      | Open WebUI (Chat)        | core       | 3000        | amd, nvidia    | Chat UI; talks to llama-server or LiteLLM. |
| dashboard       | Dashboard (Control Center) | core     | 3001        | amd, nvidia    | Operator control center, model management, service health, and setup UI. |
| dashboard-api   | Dashboard API            | core       | 3002        | amd, nvidia    | FastAPI backend for dashboard, host-agent integration, setup, models, and health. |
| model-router    | ODS Model Router         | core       | internal 9099 | all          | Stable-alias data plane. Forwards requests for `ods/current` (and compatibility aliases) to the concrete backend named by the host agent's `model-state.json`, rewriting the request model on the way in and restoring the alias on the way out. Internal-only — no host port. |
| remote-provider-egress | Remote Provider Egress | core   | internal 8091 | all          | Internal-only OpenAI-compatible egress boundary for optional remote LLM providers. Reads generated remote routing state and a private provider secret file and enforces the remote-provider egress policy. No host port. |
| remote-provider-ssh-tunnel | Remote Provider SSH Tunnel | core | internal 18090 | all      | Internal-only SSH transport supervisor scaffold for optional remote LLM providers. Reads generated remote routing state and private SSH custody files and reports a redacted fail-closed plan. No host port. |
| litellm         | LiteLLM (API Gateway)   | recommended | 4000       | all            | Unified OpenAI-compatible API gateway for local/cloud/hybrid and Lemonade paths. |
| searxng         | SearXNG (Web Search)     | recommended | 8888      | all            | Privacy-respecting metasearch for web research. |
| token-spy       | Token Spy (Usage Monitor) | recommended | 3005     | all            | Token and usage monitoring for local/proxied traffic. |
| hermes          | Hermes Agent             | recommended | internal 9119 | all          | Default generalist agent (Nous Research) with tools, memory, and skills. Not host-bound directly. |
| hermes-proxy    | Hermes Auth Proxy        | recommended | 9120       | all            | Magic-link-gated Caddy proxy in front of Hermes. |
| ape             | APE (Agent Policy Engine) | optional  | 7890        | all            | Policy/audit layer for autonomous agent tool calls. |
| brave-search    | Brave Search (Paid API)  | optional   | 8585        | all            | Optional Brave Search API bridge. Requires `BRAVE_SEARCH_API_KEY`. Opt-in searxng-compatible mode for consumers like Perplexica. |
| comfyui         | ComfyUI (Image Gen)      | optional   | 8188        | amd, nvidia    | Image generation UI and API. |
| ods-proxy     | ODS (Web)       | optional   | 80          | all            | LAN/mDNS web entry with host-based routing for chat, dashboard, API, and Hermes proxy. |
| embeddings      | TEI (Embeddings)         | optional   | 8090        | all            | Text embeddings service for RAG. |
| hipfire         | hipfire (RDNA-native LLM engine) | optional | 11435 | amd          | Rust/HIP-direct LLM inference engine for Radeon GPUs, built from source at a pinned upstream commit. Serves its own MQ4 weights (not GGUF) over an OpenAI-compatible API. AMD-only by construction — no CPU or CUDA path. Off by default (`ENABLE_HIPFIRE`); enable with `./install.sh --hipfire`. |
| langfuse        | Langfuse (LLM Observability) | optional | 3006      | all            | LLM tracing, evaluations, and prompt management. |
| model-deck      | Model Deck (VRAM Control) | optional  | 3015        | amd            | GPU/VRAM supervisor: tenant load/unload/free/park controls, config sets with preview/apply, idle-release + contention-healing arbiter. |
| n8n             | n8n (Workflows)          | optional   | 5678        | all            | Workflow automation. |
| node-agent      | ODS Node Agent (Remote GPU Metrics) | optional | 7720 | all         | Read-only, bearer-authenticated metrics sidecar deployed on **remote** inference nodes that don't run the full ODS stack. dashboard-api polls it to show that node's GPUs and what it's serving. |
| openclaw        | OpenClaw (Agents) **(deprecated)** | optional | 7860 | all | Legacy agent framework. **DEPRECATED** — removal planned in the next release. Use `hermes` instead. See [MIGRATION-OPENCLAW-TO-HERMES.md](../docs/MIGRATION-OPENCLAW-TO-HERMES.md). |
| opencode        | OpenCode (IDE)           | optional   | 3003        | all            | Host-managed browser IDE / coding assistant wired to local inference. |
| perplexica      | Perplexica (Deep Research) | optional | 3004        | all            | Deep research UI backed by SearXNG and local inference. |
| privacy-shield  | Privacy Shield           | optional   | 8085        | all            | PII detection and protection proxy. |
| qdrant          | Qdrant (Vector DB)       | optional   | 6333 / 6334 | all           | Vector store for RAG. |
| tailscale       | Tailscale (Remote Access) | optional  | host network | all           | Optional tailnet access for remote/private networks. |
| tts             | Kokoro (TTS)             | optional   | 8880        | all            | Text-to-speech. |
| whisper         | Whisper (STT)            | optional   | 9000        | all            | Speech-to-text. |

## Categories

- **core** — Always part of the base stack (llama-server, open-webui, dashboard, dashboard-api, model-router, remote-provider-egress, remote-provider-ssh-tunnel). The last three are internal-only and expose no host port.
- **recommended** — Enabled by default in the installer; can be disabled (litellm, searxng, token-spy, hermes, hermes-proxy).
- **optional** — User opts in during install or later (APE, Brave Search, ComfyUI, ods-proxy, embeddings, hipfire, Langfuse, Model Deck, n8n, node-agent, OpenCode, Perplexica, Privacy Shield, Qdrant, Tailscale, TTS, Whisper). `openclaw` is also in this category but is **deprecated** as of 2026-05-12.

This table is hand-maintained and has drifted before. The runtime source of truth is each
service's `manifest.yaml`, auto-discovered by `lib/service-registry.sh` — compare with
`scripts/validate-manifests.sh` output when adding a service.

## Ports and .env

Each service’s external port can be overridden in `.env` via the `external_port_env` field in its manifest (e.g. `WEBUI_PORT`, `OLLAMA_PORT`/`LLAMA_SERVER_PORT`). Defaults are in the table above and in `.env.example`.

The installer (phase 04) checks that these ports are free before proceeding. The service registry (`lib/service-registry.sh`) and scripts like `health-check.sh` use these ports for health checks and URLs.

## Version compatibility

All bundled extensions declare `compatibility.ods_min: "2.0.0"` (or equivalent) so that:

- `scripts/validate-manifests.sh` and `ods config validate` can report whether each extension is compatible with the current ODS version.
- Future core releases can enforce or warn when an extension’s `ods_min` is newer than the installed core, or when `ods_max` is older.

See [schema/README.md](schema/README.md) for the manifest schema and compatibility block.

## Where manifests live

Regenerate this list with `ls -d extensions/services/*/ | xargs -n1 basename | sort`.

```
extensions/services/
  ape/manifest.yaml
  brave-search/manifest.yaml
  comfyui/manifest.yaml
  dashboard/manifest.yaml
  dashboard-api/manifest.yaml
  embeddings/manifest.yaml
  hermes/manifest.yaml
  hermes-proxy/manifest.yaml
  hipfire/manifest.yaml
  langfuse/manifest.yaml
  litellm/manifest.yaml
  llama-server/manifest.yaml
  model-deck/manifest.yaml
  model-router/manifest.yaml
  n8n/manifest.yaml
  node-agent/manifest.yaml
  ods-proxy/manifest.yaml
  openclaw/manifest.yaml      # deprecated; removal planned next release
  opencode/manifest.yaml
  open-webui/manifest.yaml
  perplexica/manifest.yaml
  privacy-shield/manifest.yaml
  qdrant/manifest.yaml
  remote-provider-egress/manifest.yaml
  remote-provider-ssh-tunnel/manifest.yaml
  searxng/manifest.yaml
  tailscale/manifest.yaml
  token-spy/manifest.yaml
  tts/manifest.yaml
  whisper/manifest.yaml
```

Each directory typically also has a `compose.yaml` (and optional overlay like `compose.nvidia.yaml`). The resolver `scripts/resolve-compose-stack.sh` builds the full compose command from enabled extensions and the selected GPU backend.

## Enabling and disabling

- **During install:** Phase 03 (features) lets you enable optional features; the installer enables the corresponding extensions.
- **After install:** Use `ods-cli` (e.g. `ods enable n8n`, `ods disable comfyui`) or enable/disable by renaming `compose.yaml` to `compose.yaml.disabled` (and back) in the service directory under the install path.

The service registry only loads manifests for extensions that are “enabled” (compose file present), so disabled extensions do not appear in `sr_list_enabled` or port checks.
