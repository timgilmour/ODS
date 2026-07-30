<!-- GENERATED FILE -- DO NOT EDIT BY HAND.
     Regenerate with: python3 scripts/generate-network-ports-doc.py --write
     Sources: extensions/services/*/manifest.yaml, config/ports.json,
              config/network-exposure-policy.json -->

# Network ports

Every port a stock ODS install can listen on, with what guards it. Generated
from the manifests and `config/`, so it cannot drift from what the stack
actually deploys -- `tests/contracts/test-network-ports-doc.py` fails the build
if it does.

**Nothing here is a firewall policy.** It is the inventory you write one from.
The default posture is that these ports are reachable from the LAN if the host
firewall lets them be; services marked `operator-controlled` are expected to
stay local unless you deliberately publish them.

Reading the columns:

- **Host port** -- what is reachable on the machine. `internal only` means the
  container listens but nothing is published to the host; `host network` means
  the service bypasses port mapping and binds the host's interfaces directly,
  so a published-port table would not show it at all.
- **Env var** -- override for the host port, set in `.env`.
- **Auth** -- whether the surface authenticates at all. `no` does not mean
  broken; it means the port itself is the control, so scope it.

## Ports

| Host port | Container port | Service | Env var | Auth | Exposure | Notes |
|---|---|---|---|---|---|---|
| `80` | `80` | `ods-proxy` | — | yes | explicit | Optional web gateway for LAN/mDNS routing. |
| `3000` | `8080` | `open-webui` | `WEBUI_PORT` | yes | operator-controlled | Primary chat UI. |
| `3001` | `3001` | `dashboard` | `DASHBOARD_PORT` | yes | operator-controlled | Primary control surface. |
| `3002` | `3002` | `dashboard-api` | `DASHBOARD_API_PORT` | yes | operator-controlled | Host-agent and setup API surface. |
| `3003` | `3003` | `opencode` | `OPENCODE_PORT` | yes | operator-controlled | Browser IDE/coding assistant surface. |
| `3004` | `3000` | `perplexica` | `PERPLEXICA_PORT` | no | operator-controlled | Deep research UI backed by search and local inference. |
| `3005` | `8080` | `token-spy` | `TOKEN_SPY_PORT` | yes | operator-controlled | Usage/trace data can reveal prompts and model behavior. |
| `3006` | `3000` | `langfuse` | — | yes | operator-controlled | May contain prompts, traces, and evaluation data. |
| `4000` | `4000` | `litellm` | `LITELLM_PORT` | yes | operator-controlled | Must enforce LITELLM_MASTER_KEY when used as a gateway. |
| `5678` | `5678` | `n8n` | `N8N_PORT` | yes | operator-controlled | Workflow automation can call local and external services. |
| `6333` | `6333` | `qdrant` | `QDRANT_GRPC_PORT` | yes | operator-controlled | Vector database can contain private embeddings and metadata. |
| `7860` | `18789` | `openclaw` | `OPENCLAW_PORT` | yes | opt-in-only | Deprecated legacy agent. Must remain optional and token-gated. |
| `7890` | `7890` | `ape` | — | yes | operator-controlled | Policy/audit service for agent tool calls; keep local unless explicitly publishing an agent control surface. |
| `8085` | `8085` | `privacy-shield` | `SHIELD_PORT` | yes | operator-controlled | PII restoration/proxy surface; auth protects re-identification paths. |
| `8090` | `80` | `embeddings` | `EMBEDDINGS_PORT` | no | operator-controlled | Embedding model API; keep private unless a client explicitly needs it. |
| `8188` | `8188` | `comfyui` | `COMFYUI_PORT` | no | operator-controlled | Optional image-generation UI; should stay localhost/private by default. |
| `8585` | `8585` | `brave-search` | — | yes | operator-controlled | Bridges to a paid upstream API key; exposure risks quota/key abuse. |
| `8880` | `8880` | `tts` | `TTS_PORT` | no | operator-controlled | Text-to-speech model API. |
| `8888` | `8080` | `searxng` | `SEARXNG_PORT` | no | operator-controlled | Metasearch service; exposure can leak query history. |
| `9000` | `8000` | `whisper` | `WHISPER_PORT` | no | operator-controlled | Speech-to-text model API; audio may be sensitive. |
| `9120` | `9120` | `hermes-proxy` | — | yes | operator-controlled | Caddy forward_auth gate in front of Hermes. |
| `11434` | `8080` | `llama-server` | `OLLAMA_PORT` | no | operator-controlled | Local OpenAI-compatible inference API. |
| host network | n/a | `tailscale` | — | yes | host-network | Remote access path; host networking is intentional and must stay explicit. |
| internal only | `9119` | `hermes` | — | yes | none | Must remain internal-only; users enter through hermes-proxy. |
| internal only | `8091` | `remote-provider-egress` | — | yes | none | Internal-only boundary that injects private provider credentials and forwards remote LLM traffic. |
| internal only | `18090` | `remote-provider-ssh-tunnel` | — | yes | none | Internal-only SSH tunnel sidecar for remote LLM transport; binds no host ports and reads SSH custody files from private state. |

## Services that publish nothing

Every bundled service not listed above has no host-facing port: it is reachable
only from inside the compose network, by the services that need it.

## Adding a service

Set `external_port_default` (or `host_network: true`) in the manifest and add a
matching entry to `config/network-exposure-policy.json` -- the network-exposure
contract test already requires the pairing. Then regenerate this file:

```bash
python3 scripts/generate-network-ports-doc.py --write
```
