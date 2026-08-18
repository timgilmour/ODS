# model-deck

Model Deck is the centralized VRAM control plane for ODS, arbitrating GPU memory across a **declared set of local engines** — the fresh-install default is lemonade, ComfyUI, and hipfire, but any number of local engine instances of a known *kind* (see [Declared Engines](#declared-engines)) can be added or removed live, with no restart — with a pluggable policy system. It also manages model storage tiers (hot/cold disk moves) and node lifecycle (Spark single-slot GPU scheduling).

## Overview

Model Deck runs at `http://localhost:3015` and serves two key functions:

1. **VRAM arbitration** — policies, load/unload sequencing, tenant priority, automatic healing for idle models
2. **Storage tiering & orchestration** — cross-location model moves (hot ↔ cold disk), pull-through on load, watermark-driven eviction, multi-node GPU scheduling

The deck maintains a pure planning core (guards, feasibility checks, LRU ordering) and an imperative shell (the arbiter watcher, storage watcher, and executor threads), following the same design language as ODS system services.

## Features

- **VRAM policy** — declare how engines share GPU memory; enforce priorities and autoheal idle models
- **Set Builder** — manual load groups (e.g., "inference bundle" = model + samplers), apply with a single click
- **Storage locations** — bind-mount user drives (cold archive, secondary cache); register them with the deck; let it manage moves
- **Automatic tiering** — pull cold models to hot storage on load, evict hot models to cold when running low
- **Pull-through load** — cold models appear in the Load dropdown (marked ❄), can pull and load in one action
- **Model pins** — prevent eviction of important models; pins survive across restarts
- **Multi-node GPU** — deck orchestrates Spark (single-slot GPU node) model serving and hot-swaps
- **Lifecycle reconciliation** — records what each resource is *supposed* to be running and restores it when it dies, while never touching a deliberate park (see [Lifecycle](#lifecycle-intent-status-and-reconciliation))
- **Audit trail** — every load, unload, move, and policy change logged to `events.jsonl`

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LEMONADE_API_KEY` | *(empty)* | llama.cpp (lemonade) server API key (if required) |
| `LITELLM_KEY` | *(empty)* | LiteLLM proxy API key |
| `ODS_AGENT_KEY` | *(empty)* | Host agent API key for container control |
| `MODEL_DECK_SPARK_NODE_URL` | *(empty)* | Spark node HTTP API base URL (if using remote Spark) |
| `MODEL_DECK_SPARK_SERVING_URL` | *(empty)* | Spark serving URL (OpenAI-compatible) |
| `MODEL_DECK_SPARK_NODE_NAME` | `sparky` | Friendly name for the Spark node in the deck UI |
| `ODS_REMOTE_NODE_KEYS` | *(empty)* | JSON dict of `{node_name: api_key, ...}` for remote nodes |
| `MODEL_DECK_UPDATE_CHECK_ENABLED` | `true` | Kill switch for update-checking — see [Update checking](#update-checking) |
| `MODEL_DECK_UPDATE_INTERVAL_S` | `21600` (6 h) | Update-check cadence, on its own thread, never the watcher tick |

Settings read from the environment but **not** passed through `compose.yaml`'s
`environment:` allowlist — changing one means editing that list, not just
`.env` (Python `MODEL_DECK_*` env prefix, or defaults in `app/settings.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_DECK_STORAGE_WATCH_INTERVAL` | `60` | Watermark check cadence (seconds) |
| `MODEL_DECK_STORAGE_SLACK_BYTES` | `2e9` | Headroom to reserve when planning moves (2 GB) |
| `MODEL_DECK_LEMONADE_CONTAINER` | `ods-llama-server` | **SEED ONLY since E1** — feeds the one-time `engines[]` seed's `lemonade` connection on an upgrading box's first boot; the storage notify hook that restarts a lemonade-kind container for model registration now reads each resource's own declared `connection.container` instead (see [Declared Engines](#declared-engines)) |

## API Endpoints

### Core Status

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/state` | Live snapshot: `world` (GPUs + tenants + default route), `policy`, `models` (registry scan), `lifecycle` (intent × observation, see below) |
| `GET` | `/api/events?n=` | Audit-log tail (`events.jsonl`) |

### Tenant Control

One generic route dispatches every human-initiated action, by **verb**, over
whichever resource is named — never a fixed per-engine URL:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/tenants/{resource}/load` | Load a model (lemonade-kind only; supports `?pull=true` for cold models, `?force=true` to override the host-agent guard) |
| `POST` | `/api/tenants/{resource}/unload` | Unload the current model or a named model (lemonade-kind only) |
| `POST` | `/api/tenants/{resource}/free` | Free VRAM — drop cached state, keep serving (comfyui-kind only) |
| `POST` | `/api/tenants/{resource}/park` | Park (freeze inference, free VRAM) (hipfire-kind only) |
| `POST` | `/api/tenants/{resource}/resume` | Resume from park (hipfire-kind only) |

`{resource}` is looked up in the **live declaration** on every request — a
404 for a resource nobody declared. The verb picks the handler; a verb the
resource's declared *kind* doesn't support is a 405 naming the kind (e.g.
`POST /api/tenants/my-comfy/load` on a comfyui-kind resource). On a
fresh/default install the seeded resource names are literally `lemonade`,
`comfyui`, and `hipfire`, so the URLs above read exactly as they did before
this generalized — nothing to migrate for the common case. See [Declared
Engines](#declared-engines) for how a resource gets declared in the first
place, and which verbs each *kind* exposes.

### VRAM Policy

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/policy` | Get current policy (roles, priorities, autoheal rules) |
| `PUT` | `/api/policy` | Replace policy |

### Set Builder

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sets` | List defined load sets |
| `POST` | `/api/sets` | Create a new set (models + metadata) |
| `PUT` | `/api/sets/{id}` | Update set metadata or members |
| `DELETE` | `/api/sets/{id}` | Remove a set |
| `POST` | `/api/sets/{id}/apply` | Execute all loads in the set (atomic sequence) |
| `GET` | `/api/sets/{id}/status` | Check apply status |

### Storage Locations & Moves

See the **Storage tiering** section below for detailed semantics.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/storage/state` | Locations, catalog (units), active moves, policy |
| `POST` | `/api/storage/locations` | Register a location (requires compose mount first) |
| `PUT` | `/api/storage/locations/{name}` | Update role/watermark/archive_to/readonly |
| `DELETE` | `/api/storage/locations/{name}` | Deregister (files untouched) |
| `POST` | `/api/storage/moves` | Start a move from one location to another |
| `GET` | `/api/storage/moves/{job_id}` | Job status/progress |
| `DELETE` | `/api/storage/moves/{job_id}` | Cancel an in-flight move |
| `PUT` | `/api/storage/units/{unit_id}` | Pin a model (prevent auto-eviction) |
| `GET` | `/api/storage/policy` | Auto/manual mode, watermark defaults |
| `PUT` | `/api/storage/policy` | Set auto mode on/off |
| `POST` | `/api/storage/rescan` | Force full catalog scan |

### Nodes (registry)

See [Node registry](#node-registry-topology-credentials-and-observation) below for the full data model, seeding, and observer semantics.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/nodes` | List every registry entry + `credential_set` (never the credential itself) + `control` (declared operability: `"none"` \| `"swap"`) |
| `POST` | `/api/nodes` | Create a node-agent entry (`{id, label, address, serving_address?, credential?, control?}`). `control` is `"none"` \| `"swap"`, default `"none"` — adding a node never grants verbs implicitly. 409 on a duplicate id, 422 on an invalid slug/missing address |
| `PUT` | `/api/nodes/{id}` | Partial update — `label` / `address` / `serving_address` / `credential` / `control`. `id` is immutable |
| `DELETE` | `/api/nodes/{id}` | Remove the entry and its credential. 409 for `local` (undeletable) and for a `control: "swap"` node (demote to `"none"` first) |
| `POST` | `/api/nodes/test` | Test connection — `{node_id}` (probes with the stored credential) **or** `{address, credential}` (pre-save, e.g. before the first Save). Never both. Returns `{ok, name?, platform?, capabilities?, gpu_count?, error?}`; the credential is never echoed, including on failure |

### Declared Engines

Every node's `engines[]` — what the deck's VRAM arbitration, storage
notify hooks, and Set Builder actually watch and act on — is a **declared
list**, not a fixed lemonade/comfyui/hipfire triple. Any number of
resources of a known *kind* can be added, edited, or removed while the deck
is running; nothing restarts to pick up the change (the watcher re-reads
the declaration fresh every ~2 s tick). A fourth kind, `sglang-omni`, is
**remote-only** — declared on a node-agent entry, never on `local` — see
**Remote engines** below for what running an engine off-box actually
requires.

**Schema** — one entry per declared resource:

```jsonc
{
  "resource": "lemonade",       // unique name, no "/"; keys local/<resource>
                                 // intent + settings scopes. Immutable once
                                 // declared — see PUT's 422 below.
  "kind": "lemonade",           // one of GET /api/engine-kinds' "kinds"
  "connection": {                // shape is PER-KIND — GET /api/engine-kinds
    "url": "http://llama-server:8080",
    "metrics_url": "http://llama-server:8001/metrics",
    "container": "ods-llama-server"
  },
  "gpu_index": 1,                // which read_gpus()-filtered GPU this
                                  // resource is placed on
  "policy_defaults": {           // seeds PolicyStore on first declare
    "priority": 50, "pinned": false, "idle_ttl": 900
  }
}
```

`GET /api/engine-kinds` is the picker source: every known *kind*'s
connection schema (`{field: {required}}`), WHERE it may run
(`remote_capable` — declarable on a node-agent entry; `local_capable` —
declarable on the local one; both booleans, both enforced by the write
gate below), and its human-initiated verb vocabulary (`human_verbs`, e.g.
lemonade-kind's `["load","unload"]`, hipfire-kind's `["park","resume"]`,
sglang-omni-kind's `["load","unload"]` too — see **Remote engines** below)
— the UI never bakes a kind name in.

**CRUD** (node-scoped since this branch — `/api/nodes/{node_id}/engines[/{resource}]`
works identically for `local` and for any node-agent entry; `local` is just
an id here, not a special path segment. A resource not literally named
`lemonade`/`comfyui`/`hipfire` works identically everywhere in the deck for
observation, VRAM policy, and Set Builder/lifecycle bookkeeping — see
**Park-allowlist prerequisite** below for the one place it does NOT: a
container verb on a newly declared engine):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/engine-kinds` | Every known kind's connection schema + `human_verbs` |
| `POST` | `/api/nodes/{node_id}/engines` | Declare a new resource on `node_id`. 404 for an unknown node; 422 (one-line reason) for a shape/kind defect **or** for a kind that isn't `local_capable`/`remote_capable` for that node's `agent_kind`; 409 if `resource` is already declared on THIS node; 422 naming the owning node if ANOTHER node already declares that resource name (see **Resource names are deck-wide** below) |
| `PUT` | `/api/nodes/{node_id}/engines/{resource}` | Full-entry replace. The body's `resource` must equal the path — renaming is refused (422: "rename is refused; forget and re-add instead"), never coerced. 404 if the node or `{resource}` isn't currently declared. A KIND change forgets the resource's intent record first, so a stale record can never drive a restore through the old kind's adapter |
| `DELETE` | `/api/nodes/{node_id}/engines/{resource}` | **Forget** (see below). 404 if the node or the engine is unknown |

**Remote engines (fourth kind, first shipped: `sglang-omni`):** a kind
whose `remote_capable` flag is true may be declared on a **node-agent**
entry instead of `local` — the engine runs on that node's box, not beside
the deck (`local_capable: false` for this kind: it has no local client to
build at all, so `POST /api/nodes/local/engines` with `"kind":
"sglang-omni"` 422s at the write gate). Three things have to be true before
the declaration works end to end:

1. **The node-agent entry itself is operable** — `POST /api/nodes` (or the
   seed) recorded an `address` and a stored `credential` for it (see [Node
   registry](#node-registry-topology-credentials-and-observation) below).
   Without both, the verb route 503s: "a node-agent entry with an address,
   a stored credential and a remote-capable kind is required."
2. **The node's own `engines.json`** exists and declares the same resource
   — a small host-owned allowlist file living beside that node's
   `profiles.json` (ruling A1: the deck cannot name a compose file to run
   on someone else's box), read only by the node-agent, never written by
   it. It carries the facts only the node-agent can act on: `compose_file`
   (absolute path), `health_url`, and a `busy` probe — today exactly
   `{"kind": "connections", "port": N}`, a raw established-TCP count on the
   engine's serving port (no metrics endpoint exists to scrape instead).
   The deck-side declaration's `connection.url` is a separate,
   operator-facing record of where the engine serves for the board to
   show — the deck never dials it directly, only through the node-agent's
   channel — so the two files can in principle disagree, and only
   `engines.json` governs what actually launches.
3. **The resource name is unique across the whole deck**, exactly as
   **Resource names are deck-wide** below already states — a remote
   declaration is not a separate namespace.

Acting on a declared remote engine is a **separate route** from the local
`/api/tenants/{resource}/{verb}` dispatcher (above), because it goes over
the node-agent's own up/down channel rather than the deck calling the
engine's HTTP API directly:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/nodes/{id}/engines/{resource}/{verb}` | Act on a node-agent-declared engine (today: `load`/`unload`). **202**, not 200 — the node-agent queues the request for the host-side swap-helper and this call does not wait for or observe the result (a cold sglang-omni start measures ~3.5–4.5 min, GF4). 404 unknown node or resource; 405 the kind doesn't support `verb`; 501 the kind's vocabulary includes `verb` but the node-agent channel has no call for it (only `load`/`unload` map to the channel's `up`/`down` today); 503 the node isn't operable (point 1 above) |

Intent is recorded **before** the call, the same rule as every long-running
actuator elsewhere in this doc, and rolled back (a compare-and-swap on the
exact record this call wrote) if the POST itself raises, so a request that
never left the deck records nothing durable. State comes back through the
normal observation path — `GET /api/state`'s `lifecycle` block, keyed
`<node>/<resource>` — never from a result file: the node-agent's own
`engine-status-<resource>.json` is forensics for a human reading logs on
the box, not read back by anything here.

**Boot-tail note.** A cold sglang-omni boot the deck itself started can
show the board reading `quarantined; awaiting operator` for roughly the
last ~3 minutes of the ~3.5–4.5 minute cold start (GF4) — this is expected,
not a stuck engine, and specific to the *reconciler's automatic* restore of
an engine that died out-of-band: that restore deliberately does not
re-stamp intent (so a genuinely crash-looping engine can still reach
quarantine), so once the 2-consecutive-failure budget is spent — about 60 s
after the reconciler dispatched the restore, well before the container is
actually healthy — the resource reads quarantined while `docker compose up
-d` keeps running underneath, unaffected. It **self-heals on the first
`serving` tick**: the moment the engine reports healthy, the intent
store's success path clears both `quarantined` and the failure count. An
*operator*-initiated Load (the route above, which stamps a fresh intent
timestamp) is fully covered by the kind's own ~10-minute warming window
instead and should never reach quarantine within a normal boot.

**Forget semantics (bookkeeping only):** `DELETE
/api/nodes/{node_id}/engines/{resource}` drops the declaration entry, the
resource's intent record, and its stored policy row — and **nothing
else** (settings scopes, provenance, and events all survive, same posture
as node removal). It **never calls the engine** — no client lookup happens
anywhere in the route. A still-running container or model is left exactly
as it was; the deck simply stops watching and stops arbitrating it. To
actually stop the underlying process, do that separately (e.g. `docker
stop`) before or after forgetting the declaration — the two are
intentionally decoupled.

**Resource names are deck-wide:** a resource name may be declared on
exactly ONE node. The write gate refuses a name another node already
declares (422 naming both the resource and the owning node), and a
hand-edited `nodes.json` holding a duplicate heals on load — the first
entry in file order keeps the name, the later one loses that one engine
and keeps the rest. This is what lets policy rows stay keyed by bare
resource (`policy.json` has no node dimension): a row unambiguously belongs
to one declaration, so forgetting an engine on any node forgets *its* row.
Intent and observation stay node-keyed (`<node>/<resource>`) regardless — a
key still has to say which box.

**Park-allowlist prerequisite (container verbs only):** declaring a
resource is enough for the deck to *observe* it, apply VRAM policy to it,
and include it in a Set — but a container-affecting verb (hipfire-kind's
`park`/`resume`; the storage pull-through hook's automatic restart of a
lemonade-kind resource after a moved-in GGUF) additionally requires the
resource's declared `connection.container` name to be in
`settings.park_allowlist` — env `MODEL_DECK_PARK_ALLOWLIST`, default
`["ods-hipfire", "ods-comfyui", "ods-llama-server"]` (`app/settings.py:95`).
`DockerCtl` refuses any other name with `GuardError`, checked BEFORE any
HTTP call is made (`app/docker_ctl.py:198-199`) — so a newly declared
engine whose container isn't allowlisted 409s on park/resume, and the
storage notify hook logs (and isolates, never aborting a sibling resource's
own restart) a `notify-restart-failed` event for it instead of registering
the moved-in model. `load`/`unload`/`free` are unaffected — they reach the
engine's own HTTP API directly, never Docker.

There is a SECOND, deploy-level gate behind the one above: the
`ods-docker-ctl` socket-proxy sidecar only forwards the Docker Engine API
methods its `compose.yaml` command explicitly allowlists by path regex
(`-allowGET`/`-allowPOST`). Today `GET .../json` and
`POST .../{start,stop}` are wildcarded across every container name, so once
a container clears the app-level allowlist above, start/stop already work
with no compose edit — but an engine kind that ever needed a *different*
Docker API verb (`exec`: deliberately absent from the proxy's rules
entirely today, since C2 removed its only caller — see `compose.yaml`'s
`ods-docker-ctl` service comment) would need its own compose-level rule
added too, name-PINNED rather than wildcarded (the same comment's stated
reasoning: `exec` runs an arbitrary command as root, so a wildcard there
would make the in-process allowlist the only thing standing between a bug
and host-wide RCE) — on top of the settings-level allowlist entry above.

**Coexistence (binding invariant):** the deck never actuates anything
without first recording *intent* for it (see [Lifecycle](#lifecycle-intent-status-and-reconciliation)).
A model or container that something *else* started — ODS-native, a human
at the CLI, a script — is observed (it shows up in `/api/state`'s
`world.tenants`) but is left completely alone: no intent means no restore,
no eviction candidate consideration beyond the normal VRAM-contention
rules any loaded resource is subject to, and no assumption that the deck
did it. Declaring an engine is additive (existing engine state is
unaffected — nothing loads or restarts just because you added the
declaration), and forgetting one is subtractive in the same way (nothing
unloads or stops just because you removed it). The declared set is the
deck's *scope of attention*, not a lever on the engines themselves.

### Serving (node-addressed swap control)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/nodes/{id}/serving/status` | The node's status (profiles, swap status, what is serving). 404 for an unknown node id, 503 if the node is not operable (`control` != `"swap"` or a prerequisite missing) |
| `POST` | `/api/nodes/{id}/serving/swap` | Hot-swap the node's single slot to a profile (`{profile, force}`). 404 unknown node, 503 not operable |
| `POST` | `/api/nodes/{id}/serving/reload` | Ship the resolved DECLARED-only settings for whatever profile is (or will be) serving to the node, then re-swap that same profile so they actually launch (`{profile?, force?}` — no `profile` reloads whatever last swapped in). The one human action design decision 5 calls for; see [Settings](#settingsjson--what-things-are-launched-and-served-with) below. 404 unknown node, 503 not operable |

### Removed routes

- `/api/spark/*` (status/swap/reload) — the node-less alias for
  `/api/nodes/{id}/serving/*`, kept for one deploy cycle after N1 shipped
  and removed 2026-08-13 as pre-registered here. A 404 on
  `GET /api/spark/status` — or a 405 on the two POSTs, from the UI
  bundle's static catch-all, which owns every unrouted path but serves
  only GETs — means you are on the removed alias: same bodies/semantics
  live at the serving routes above, addressed by node id.

### Lifecycle

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/state` | The `lifecycle` block: per-resource `{status, reason, intent, observed, last_healthy_ts}` |
| `POST` | `/api/lifecycle/quarantine/{key}/clear` | Release a quarantine so the reconciler will try again (404 if the key has no intent) |
| `POST` | `/api/lifecycle/adopt/{key}` | Record what is *already* running as the intent. Bookkeeping only — never loads, unloads, or restarts anything |
| `GET` | `/api/lifecycle/auto` | Whether the reconciler may act |
| `POST` | `/api/lifecycle/auto` | Turn automation on/off (`{"enabled": bool}`, strict). Off stops the Deck acting; it unloads nothing |

`{key}` is a resource key and contains a slash (`local/hipfire`), e.g.
`POST /api/lifecycle/adopt/local/hipfire`.

### Settings & rename planning

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/settings/catalog/{node}/{engine}` | The engine's harvested option catalog (`null` if it has never been up) |
| `GET` | `/api/settings/effective/{node}/{engine}/{model}?layers=` | Resolved ladder for one placement — `resolved` (all five layers, full provenance) and `argline` (declared-only, what would actually ship); `?layers=` filters `resolved` only |
| `POST` | `/api/settings/preview` | Parse a typed argline into a settings map without saving — the text field's live feedback |
| `POST` | `/api/settings/adopt/{node}/{engine}` | Sweep the node's real compose profiles into the settings store (see **Adopt**, below). Only `(<node>, "vllm")` is adoptable, and only for a `control: "swap"` node |
| `GET` | `/api/settings/{kind}/{key}` | One scope (`kind` is `engines`, `models`, or `engine_models`) |
| `PUT` | `/api/settings/{kind}/{key}` | Merge values into one namespace of one scope: `{"namespace": "args"\|"env"\|"container", "values": {...}}` |
| `POST` | `/api/rename/plan` | Read-only: plan the alias → identity rename migration for the single control:"swap" node's vLLM profiles (`{"client_pins": {route: [pin, ...]}}`, optional). Plans, never executes — see `~/notes/model-deck-rename-runbook.md` |

### `characteristics.json` / `declared.json` — model and engine facts

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/facts` | Every known key's resolved facts (declared-over-derived, provenance intact) |
| `PUT` | `/api/facts/declared/{key}` | Set declared fields for `{key}` (allowlist only; 422 on a derivable field, a malformed key, or an empty body) |
| `GET` | `/api/facts/drift` | Per-key drift report — facts that should agree but don't (see below) |

**Derive, don't duplicate.** Every field has exactly one authoritative
source. `characteristics.json` is a machine-owned cache of facts read from
the things that own them (a checkpoint's `config.json`, an engine's
`/v1/models`), every field stamped with `source` and `derived_ts`.
`declared.json` is a small human-owned allowlist for what cannot be derived:
`tools_verified`, `label`, `notes`, `tags`, `engine_preference`.

Resolution is declared-over-derived, with the derived value retained as
`shadowed_value` so the UI can show that a human overrode a reading.

**A model's identity is its checkpoint directory name, verbatim.** Aliases
are retired; role names like `fast`/`deep` are `tags`.

**Drift is reported, never corrected.** `GET /api/facts/drift` compares
facts that should agree — the checkpoint's `quant_method` against the
profile's `--quantization` (severity `crash`; this is the modelopt loop),
and served context against both the checkpoint's capability and what the
gateway advertises. Absence of a fact means "cannot check", never
"mismatch".

### `settings.json` — what things are launched and served with

Three scopes (`engines` keyed `<node>/<engine>`, `models` keyed by identity,
`engine_models` keyed `<node>/<engine>|<model>`) × three namespaces (`args`,
`env`, `container`). Resolution is a five-layer ladder — engine defaults →
checkpoint recommendations → engine → model → engine×model — merged **per
key**, most specific winning. `None` at a higher layer unsets a lower one
(the absence of `--quantization` is a real, correct configuration).

The chip panel and the free-text field are **two views of one store**;
`app/argline.py` guarantees the round trip, and unknown tokens survive it.

**Validation warns and never blocks** — unknown flag, type/choice violation,
or conflict with a derived fact (`--quantization modelopt` against a
`compressed-tensors` checkpoint: severity `crash`).

**Saving changes intent only.** A loaded placement gains `settings_drift`
with the changed keys (namespace-qualified, `"args:max-model-len"` never a
bare `"max-model-len"` — same-named keys in different namespaces must stay
distinguishable); the reconciler never restarts on it. Reload is a human
click, and it's the ONLY thing that clears `settings_drift`: reloading
re-records the placement's intent, and the drift flag's baseline *is* that
intent's `updated_ts` — so re-recording it is the whole clearing mechanism.
Nothing corrects drift by itself; a save that nobody reloads stays flagged
forever, on purpose.

Option catalogs are harvested by **argparse introspection inside the running
engine container** (`vllm serve --help` crashes without a GPU, and its text
carries no types or choices). No catalog is a supported state.

`container` settings are an allowlist: `image`, `shm_size`, `ulimits`.
Volumes stay Deck-managed — the model mount *is* the placement.

#### Applying settings — one mech per engine capability

`app/configure.py`'s `apply_settings(mech, ...)` is the one dispatch point
between "settings saved" and "an engine actually sees them." The write
boundary is what each engine's capability descriptor declares, not
local-vs-remote:

| Mech | Engines | Behavior |
|------|---------|----------|
| `api` | lemonade | A live call; applies immediately, no reload needed |
| `env+restart` | hipfire, comfyui | Writes environment; reports `requires_reload` — a restart applies it later |
| `node-settings` | spark (and future remote engines) — **implemented, Plan C2** | Ships a settings *document* to the node-agent; the host-side swap-helper merges it into the next launch. Always `requires_reload`: applying it is the human's Reload click, never this call |
| `none` | anything the Deck doesn't own | Read and warn, permanently — keeps the general rule honest for a source it cannot configure |

**Nothing here restarts anything.** A save changes intent; the reload that
applies launch-class settings is always a human click. Applying an *empty*
settings map is a no-op, never a wipe — "I have nothing to say about this
engine" must not mean "clear its configuration."

#### `node-settings`: document → override → helper-owned launch

For spark, "shipping settings to the node" means writing a small JSON
*document* — `{"args", "env", "argv", "service"}`, `argv` and `service`
pre-rendered by the Deck (the shared `render_argv`/`_declared_only` code
path both `GET /api/settings/effective/...` and
`POST /api/nodes/{id}/serving/reload`
use, so the two can never disagree about what "declared-only" ships) — to
a file the node-agent and the privileged host-side swap-helper share.
Node-agent has no docker access at all; it only ever writes
`<settings-dir>/<profile>.json`, atomically. The swap-helper reads that
document at the *next* swap of that profile, renders it into a small
compose **override** file (`command:` / `environment:` for the one
service), and launches with both files: `docker compose -f
compose-<profile>.yaml -f settings-<profile>.override.yaml up -d`.

**The Deck never rewrites `compose-*.yaml`.** The override is a separate
file, regenerated fresh from the settings document on every launch that
has one; the base compose file — and the human-authored comments in it —
is never touched. A missing, empty, or unusable settings document falls
back to launching the base compose file exactly as `swap.sh` always did —
a settings bug can degrade a launch to "as configured in compose," never
break it.

#### Adopt: importing what's already running

`POST /api/settings/adopt/{node}/{engine}` (today: only `sparky/vllm`)
sweeps the node's real `compose-*.yaml` profiles and imports each one's
`command:`/`environment:`/container fields into the matching
`engine_models` scope, plus any inline comment explaining an absent flag
(carried through as a `note` — the reason `--quantization` is missing from
heretic's launch would otherwise live only in a compose comment nobody
reads before regenerating it). **Never clobbers**: a scope that already has
something in its `args` namespace is left untouched and reported `kept`,
not overwritten, so re-running adopt after a human has already edited a
profile is safe. A profile that isn't vLLM (`ds4`, `comfyui`, ...) is
reported `skipped` with its real engine named, never guessed at. Adopt also
records the profile → identity map (which compose service and container
each profile is) as a characteristics field — that map is what lets
`settings_drift` and `POST /api/nodes/{id}/serving/reload` translate a
spark *profile*
(what intent records) into an `engine_models` scope key (what settings are
keyed by) without re-parsing compose themselves.

## Storage tiering (hot/cold model moves)

Model Deck extends GPU memory management one tier down the memory hierarchy: today the deck arbitrates **VRAM ↔ hot disk**; storage tiering adds **hot disk ↔ cold disk** (e.g., SSD cache ↔ external archive drive) with the same design language — policy, watcher, pure planner, pins, priorities, audited actions.

### Two-step location declaration

Each storage location needs two setup steps (both are idempotent):

1. **Compose bind mount** (in `compose.yaml`, under the `model-deck` service):
   ```yaml
   volumes:
     - /mnt/cold/models:/stores/cold:z        # cold archive drive
     - ./data/models:/stores/gguf-hot:z       # rw view of the GGUF store
   ```

2. **Registration** via the UI (Storage tab) or `POST /api/storage/locations`:
   ```json
   {
     "name": "cold",
     "path": "/stores/cold",
     "role": "cold",
     "store_type": "gguf",
     "engine": "none",
     "watermark_gb": null,
     "archive_to": null,
     "readonly": false
   }
   ```

An **unregistered mount does nothing**; only registered locations appear in the deck's catalog and storage state.

**Important:** When registering the `/gguf-store` mount as managed, change its `compose.yaml` mount from `ro` (read-only) to `rw`:
```yaml
# Before (unmanaged):
- ./data/models:/gguf-store:ro,z
# After (registered and managed):
- ./data/models:/gguf-store:rw,z
```
The `/gguf-store:ro` mount remains functional for backward compatibility (lemonade and ComfyUI still discover it), but only a registered location can be written to by the mover.

### Marker files: unavailable ≠ empty

On location registration, the deck writes `.deck-store.json` at the location root (containing the location UUID and name). Every scan verifies the marker file:

- **Marker present and valid** → location is `available` (files can be read and written)
- **Marker missing or invalid** → location is `unavailable` (a distinct state from empty)

An unavailable location (e.g., an unplugged cold drive) is treated with extreme care:

- **Catalog entries for unavailable locations are retained** — unplugging a cold drive does not make its models "vanish" or trigger phantom re-pulls or garbage collection
- **The deck never writes into an unavailable location** — a forgotten mount can't silently fill the container overlay or shadow-write into an empty mountpoint directory
- **Manual moves to unavailable locations are refused** with a clear "drive unmounted?" message

**Registration before mounting fails safe:** if you register a location before mounting it in compose, the path check fails first (GuardError "does not exist — is the drive mounted into the container?"), and registration is rejected. No orphan entries are created.

### Pull-through on load

Cold models appear in the lemonade Load dropdown marked with a ❄ (snowflake) indicator, showing they are not currently on hot storage. Selecting a cold model triggers a two-stage job:

1. **Pull** — move the model from its cold location to the engine's primary hot location (chosen by free space: the hot location with the most available bytes wins)
2. **Load** — once the pull completes, lemonade is notified and the model is loaded

#### The `?pull=true` contract

The pull-through feature respects two modes:

- **`auto: on`** (in storage policy) — cold models pull and load automatically with one click; no confirmation
- **`auto: off`** — cold models require explicit confirmation: `POST /api/tenants/lemonade/load?pull=true` (the UI offers a "pull and load?" dialog)

Without `?pull=true` in manual/off mode, loading a cold model fails with a clear message:
```
model 'qwen-70b' is cold (in 'archive') — re-request with ?pull=true to pull it to hot storage first
```

#### Deferred-restart caveat (lemonade only)

After a pull completes, the deck must notify lemonade that a new model file has arrived. Since **lemonade registers GGUF files only at startup** (no rescan endpoint), the notification hook restarts the lemonade container.

**If a model is already loaded in lemonade, the restart is deferred with a visible warning:**
> "lemonade has a model loaded — restart deferred; the new file registers after the next lemonade restart"

The pull job succeeds (files are moved), but no automatic load is attempted. The operator can:
- Unload the current model (via `POST /api/tenants/lemonade/unload`), which clears the way for the restart, then retry the load
- Wait for the idle-unload TTL to trigger (if configured), which has the same effect
- Issue a new load after a manual restart: `docker compose restart ods-llama-server` followed by the load

**ComfyUI and `engine: none`** locations have no restart caveat: ComfyUI lists files per-request, so no action is needed; plain locations need no engine notification.

### Watermark eviction (auto-tiering)

A **watermark** is a minimum free-space target on a hot location. When `free_bytes < min_free_gb × 1e9`, the watcher enqueues eviction actions:

- **Candidates:** non-pinned, non-loaded, non-default-route models on that location, sorted LRU by `last_used`
- **Archive-to:** the location's declared `archive_to` destination (e.g., hot location archives overflow to `cold`)
- **Partial relief:** if the archive can't accommodate all candidates, it archives what it can and reports a `storage_shortfall` event

The watcher ticks every 60 seconds, respects the `auto: on/off` toggle, and yields to any in-flight actuator (an arbiter tick's actuation phase, a set apply, or the pull-through completion hook — `apply_in_progress()` peeks the one shared lock in `app/actuation.py`) and to active move jobs.

**`last_used` tracking:** `last_used` is updated on every load the deck itself performs — a manual load (`POST /api/tenants/lemonade/load`), a pull-through load of a cold model, a config-set apply's `load_lemonade` step, and the arbiter's contention-heal reload. Models that have never been loaded through the deck (including any loaded out of band, straight against lemonade) keep `last_used = null`, are evicted first, and tie-break among themselves by filesystem mtime.

**Not implemented:** the deck does not observe litellm *default-route changes* — becoming the default route does not itself touch `last_used`. Nothing unsafe follows from that: the current default-route model is exempt from eviction outright, whatever its `last_used` says.

### The auto toggle

Storage policy includes a global `auto: on | off` flag:

- **`auto: on`** — watcher silently archives excess models (per watermark rules); pull-through loads happen automatically
- **`auto: off`** — watcher only *suggests* via events (`storage_suggestion`); pull-through requires `?pull=true` confirmation

Regardless of the mode, manual moves (drag-and-drop in the UI, `POST /api/storage/moves`) are always available.

### Example: two-location setup (hot cache + cold archive)

Here's a complete compose.yaml snippet for the model-deck service, showing typical hot/cold binding:

```yaml
# --- Storage tiering (optional) --------------------------------------
# Each storage location needs a rw bind mount here + a registration in
# the deck UI (Storage tab) or POST /api/storage/locations. Examples:
#   - /mnt/cold/models:/stores/cold:z        # cold archive drive
#   - ./data/models:/stores/gguf-hot:z       # rw view of the GGUF store
#     (register as role=hot, store_type=gguf, engine=lemonade; the
#     read-only /gguf-store mount above stays — registry keeps using it)
# An unregistered mount does nothing; an unmounted registered location
# shows as "unavailable" and is never written to (marker-file check).
```

A practical configuration:

- **Compose mount:** 
  ```yaml
  volumes:
    - ./data/models:/stores/hot:z           # SSD: ~500 GB
    - /mnt/cold/models:/stores/cold:z       # HDD: ~10 TB
  ```

- **Registrations (via UI or API):**
  - Location `hot`: role=hot, store_type=gguf, engine=lemonade, watermark_gb=50, archive_to=cold
  - Location `cold`: role=cold, store_type=gguf, engine=none, watermark_gb=null, archive_to=null

- **Behavior:**
  - Load a model from `hot` → it stays there (already hot)
  - Load a model from `cold` → pull to `hot` (auto or `?pull=true`), then load
  - Hot store drops below 50 GB free → watcher archives oldest-used models to `cold` (if auto=on)
  - Unplug the cold drive → all catalog entries survive, but moves to `cold` are refused ("unavailable")
  - Replug the cold drive → deck sees the marker file, location becomes available again, moves resume

## Lifecycle: intent, status, and reconciliation

The bug this feature exists to kill: **a deliberately parked engine and a dead one produce identical observations.** `ods-hipfire` sat `Exited(0)` for 26 hours on 2026-08-03 while the deck displayed it exactly as it displays a park, and nothing anywhere said otherwise. Status therefore cannot be read off an observation alone — it is a function of the observation **and** of what the deck intended.

### `intent.json` — desired state (`/data/intent.json`)

A flat `{"<node>/<resource>": record}` mapping recording what each resource is *supposed* to be running:

```json
{
  "local/hipfire": {
    "state": "loaded",
    "model": null,
    "engine": "hipfire",
    "actor": "operator",
    "updated_ts": "2026-08-04T09:12:44.117034+00:00",
    "last_healthy_ts": "2026-08-04T09:31:02.550881+00:00",
    "failures": 0,
    "quarantined": false
  }
}
```

- Keys are `<node>/<resource>`: `local/lemonade`, `local/hipfire`, `local/comfyui`, `sparky/slot0`. There are **no known-key defaults** — keys are discovered at runtime, which is what lets a new engine or node work without a code change, and a missing file is legitimately empty rather than "needs materializing".
- `model: null` means *"loaded, no opinion which model"* — the correct record for single-model engines like hipfire, whose model the deck does not choose. Recording a name the deck cannot observe would manufacture permanent drift. For `sparky/slot0` the recorded identity is the **profile**, not the served model name (mm27b serves under `--served-model-name aeon`).
- `state: "unloaded"` is intent, **not** an absence of it. A deliberate park is a recorded decision and the reconciler will never undo one. Deleting a key (`IntentStore.forget`) is the only way to say "the deck has no opinion".
- `actor` (task 6 follow-up) is `"operator"` or `"deck"` — who authored the record, not what it says. Every router-initiated record (control routes, a config-set apply's goal-intent recording, spark swap, park/unload) is `"operator"`, including the pull-through hook's own completion record — it is completing an operator's earlier request, just minutes later. Only the arbiter's own two automatic records — idle-release/contention-eviction unload and pending-load retrigger — are `"deck"`. A record with no `actor` at all (pre-upgrade `intent.json`) reads as `"operator"`, the conservative default. This is what lets the pull-through hook's supersession check (below) tell an operator's later action apart from the deck's own automatic churn: an idle model unloading mid-copy must not silently drop an operator's explicit pull-through load.
- Intent is recorded implicitly, on every deliberate action, guards first — a guard-refused action never happened, so it records nothing. Beyond that, **whoever actuates, records**, and *when* depends on how long the call can run: lemonade load/unload (including the deferred pull-through load, minutes later) and the watcher's own idle-release/contention-evict/load-retrigger record **before** the engine call, so a tick landing mid-call — or a call that itself fails — still sees the stated intent, not stale state (a failed lemonade load/unload is retried under the failure budget, not left unrecorded). hipfire park/resume, spark swap, and every *completed* step of a set apply record **after** the call returns: their guard refusals raise inside the client call itself, so reaching the record already means it succeeded.
- **Pull-through supersession exception:** the completion hook's own load (the deferred lemonade load above) is skipped outright — recording nothing, restarting nothing — if the *current* intent is an **operator**-authored record whose `updated_ts` postdates the pull's submission (`app/routers/control.py`, `_pull_through`/`after`). A deck-authored record (automatic idle-release/eviction) never triggers this skip, only an operator's.
- Writes are atomic (temp + `os.replace`); a missing or corrupt file reads as `{}`.

### Status vocabulary (`app/lifecycle.py`)

`derive_status(intent, observed)` is a pure function returning `{status, reason}`. Reachability is checked first, then a boot in flight, then intent × observation:

| Status | Intent | Observed | Meaning |
|---|---|---|---|
| `serving` | loaded | loaded, same model (or `model: null`) | as intended |
| `drifted` | loaded | loaded, **different** model | something else is running here |
| `down` | loaded | not loaded, node **reachable** | it died — **the only actionable status** |
| `parked` | unloaded | not loaded | deliberate. Never restored |
| `unexpected` | unloaded | loaded | somebody else started it |
| `unmanaged` | *none* | loaded | running, no intent — an adopt candidate |
| `idle` | *none* | not loaded | no intent recorded, nothing loaded |
| `unreachable` | any (retained) | node did not answer | **we failed to look.** Not evidence of anything |
| `quarantined` | loaded | not loaded, budget spent | restores failed repeatedly; awaiting an operator |
| `warming` | any | a load/boot in flight | transient; never actionable |

Two ordering rules that are easy to get wrong:

- **`unreachable` beats everything.** An engine whose status call *raised* was not observed to be unloaded — we failed to observe at all. Calling that "not loaded" would let the reconciler restore something that is already running. (This is the storage feature's `unavailable ≠ empty` rule, one level up.)
- **Quarantine is checked *after* a healthy match.** A quarantined resource that is nonetheless serving reports `serving`: quarantine describes our restore attempts, not reality.

`app/observe.py` is the single place per-engine vocabulary is translated into the one shape `{reachable, loaded, model, transitioning}`. Adding an engine means adding a mapping there, not editing status derivation, reconciliation, or the API. Notable mappings: a tenant degraded to `"unknown"` by `World.snapshot` becomes **unreachable**, hipfire's `"loading"` becomes **transitioning** (→ `warming`), and ComfyUI's `"idle"` still counts as **loaded** (it holds VRAM between jobs).

### The reconcile pass

`Watcher._reconcile_pass` runs at the **end of every watcher tick, after arbitration, on the same snapshot** — deliberately. Arbitration settles VRAM contention happening right now; reconciliation settles desired state over time. The other order can restore a model that arbitration is about to evict (a load/evict flap).

Both arbitration and this reconcile pass only run once the tick's actuation phase has acquired `app/actuation.py`'s single process-wide lock — the SAME lock a config-set apply and the pull-through completion hook hold. Exactly one of the three actuates real engine state at a time; the tick try-acquires and skips a whole actuation+reconcile pass cleanly when it's busy elsewhere. Worst-case waits for the other two: an apply through `hostagent.activate` can block up to ~600 s (`app/engines/hostagent.py`'s read timeout); the pull-through hook can hold it up to ~285 s (a lemonade container restart + its readiness poll + a lemonade load).

`plan_reconcile` (pure, `app/reconcile.py`) emits an action for exactly **one** status: `down`. Every other status is inert, and each refusal is a real incident rather than caution for its own sake — restoring a `parked` resource fights the operator every tick; auto-correcting `drifted`/`unexpected`/`unmanaged` means acting on state the deck did not author; retrying a `quarantined` key is the crash loop the budget exists to stop; `unreachable` is a node being off, not a model having fallen over; `warming` is a boot whose "not loaded yet" is indistinguishable from "died".

Restores dispatch by engine: hipfire resumes its container, lemonade loads by name, spark swaps a profile. **ComfyUI is deliberately absent** and cannot reach the dispatcher — a dead ComfyUI derives `unreachable`, never `down`.

A `serving` status stamps `last_healthy_ts` (`IntentStore.note_healthy`), which is what turns "down" into "down *since when*".

### Failure budget and quarantine

A restore counts as failed only when it **raises**. After `FAILURE_BUDGET` (2) consecutive failures the key is quarantined and the reconciler stops touching it, logging `lifecycle-restore-failed` and then `lifecycle-quarantined`.

Three things release a quarantine:

1. `POST /api/lifecycle/quarantine/{key}/clear` — the operator's "try again".
2. A successful observation (`note_healthy`, on any `serving` tick).
3. **Any new deliberate load or unload** — `IntentStore.record()` resets `failures`/`quarantined` (decided 2026-08-04). A deliberate action is evidence the situation changed (backend fixed, VRAM freed, different model chosen), so the resource earns a fresh budget. Leaving the flag set would exclude it from automatic restore forever *and invisibly*: `derive_status` only reports `quarantined` on the loaded-intent branch, so a quarantined-and-parked resource hides the flag while still being permanently excluded. `record()` preserves `last_healthy_ts` — that is the resource's health history, unrelated to what the operator now wants of it.

### Adoption

`POST /api/lifecycle/adopt/{key}` records the resource's *current observed* state as its intent, turning an `unmanaged` resource into a managed one. It changes bookkeeping only and must never load, unload, or restart anything — adoption that actuates would make "start managing this" a dangerous button, and nobody would press it.

Adopting an **unreachable** resource is refused with **409**: an observation we failed to make is not evidence, and the record it would write (`state: "unloaded"`) is a park nobody asked for — after which the reconciler would correctly refuse to restore it forever.

### The automation toggle

Auto-restore is **on by default** (unlike storage tiering, whose automation moves bytes and defaults off): lifecycle auto-restore only returns a resource to a state the operator already chose, and its absence is what let hipfire stay dead for 26 hours. `plan_reconcile` returns nothing at all when it is off.

The toggle lives in the reserved `_auto` key of `policy.json` (`{"_auto": {"enabled": false}}`), read via `PolicyStore.auto_enabled()`.

Control it over HTTP:

```
GET  /api/lifecycle/auto     ->  {"enabled": true}
POST /api/lifecycle/auto     <-  {"enabled": false}
```

It needs its own route because `PUT /api/policy` deliberately *rejects* `_auto` as reserved, so the toggle cannot ride on the tenant policy payload. `enabled` is a `StrictBool`: `"yes"` or `1` are refused with a **422** rather than guessed at, because a brake should never interpret an ambiguous value.

**Turning it off does not unload anything.** It stops the Deck acting on its own and leaves every resource exactly where it is. There is still no UI control — the whole lifecycle surface is curl-level today.

### Audit events

`lifecycle-restore` · `lifecycle-restore-failed` · `lifecycle-quarantined` · `lifecycle-spark-unreachable` (deduped — sparky is normally off).

### Spark observation cost

`SparkClient.status()` costs two node-agent requests, and the watcher ticks every 2 s. When sparky is off — its normal state — each of those blocks on a 5 s HTTP timeout, which would stretch the arbiter's cadence to ~12 s exactly when nothing is wrong. `app/observe.py::SparkObserver` therefore TTL-caches the observation (10 s) and backs off exponentially on failure (15 s → 300 s cap), and one instance is shared by the watcher and the HTTP paths, so a tick and a `GET /api/state` in the same second cost one probe. A swap invalidates the cache. A probe failure reads as **unreachable**, never as "nothing loaded", and is parked for the watcher to log once (it owns the audit trail).

### Test coverage, and one honest gap

Unit rows cover intent, status derivation, reconcile planning, observation, the routes, and the watcher pass. Live drills (`livetests/test_disruptive_lifecycle.py`, disruptive tier):

- **D7 — out-of-band stop is restored.** Stop `ods-hipfire` behind the deck's back; it must read `down` (never `parked`) and come back unaided. **Passes live.**
- **D9 — a deliberate park stays parked.** THE regression: park hipfire, watch ~30 reconcile ticks, nothing may touch it. **Passes live.**
- **D8 — quarantine after two failed restores. SKIPPED, and not passing.** There is no live-safe failure injection on this box: a restore only counts a failure when it *raises*, and `DockerCtl.start()` returns 204 even for a container that starts and instantly exits; a removed or renamed container makes `status()` raise and therefore derives `unreachable`, never the actionable `down`; the one remaining lever (stopping `ods-docker-ctl`) would disable the very restore path under test; and sparky, the other natural source of failing restores, is powered off. **The failure budget is proven by unit tests only.** What *is* asserted live is the release route (D7 exercises the quarantine-clear endpoint).

### Safety invariants (18–21): lifecycle

**18.** A deliberate park is never undone. `parked` produces no action, ever — the single most important invariant in the lifecycle work (drill D9).

**19.** `down` is the only status that acts. State the deck did not author (`drifted`, `unexpected`, `unmanaged`) is reported, never corrected.

**20.** A failed observation is never treated as an absence: `unreachable` retains the last-known intent and is not actionable, and adoption of an unreachable resource is refused (409).

**21.** A resource is restored at most `FAILURE_BUDGET` (2) consecutive times before it is quarantined and left alone for an operator.

### Known lifecycle gaps

- LiteLLM's `/health` payload is **not consumed** by status derivation. An
  `interpret_health` helper (the "not-loaded ≠ down" reading — a connection
  error means DOWN, a model-not-found error on a reachable node means NOT
  LOADED and is never an alarm) was built for it, sat dead for months, and
  was deleted in the 2026-08-12 simplify sweep (recoverable from git).
  Local engines are observed directly, which is strictly better
  information; interpreting `/health` for *remote* routes waits for the
  multi-node work that makes a route's node knowable.
- `observe_local` names the three current tenants explicitly. That is the adapter's job (it is the vocabulary boundary), but a fourth local engine does touch that file.
- **There is no lifecycle UI.** The `lifecycle` block is served and typed (`ui/src/api.ts`), but no component renders status, quarantine release, or adopt — those are curl-level operations today.

## Node registry: topology, credentials, and observation

Closes the last deferred piece of the model-centric Deck rework: node
topology (which boxes exist, how to reach them) used to live entirely in
`MODEL_DECK_SPARK_*` env vars, one hardcoded remote node, no way to add a
second one without editing compose. `app/node_store.py` makes topology and
credentials **data**, added/removed/edited through the API and the Nodes
tab, with the local box always present as a registry entry like any other.

v1 is deliberately **observe-only for anything beyond local + spark**: a
newly added node shows up with reachability, GPU, and serving telemetry —
no verbs, no placements, no lifecycle. Full engine operability (swap,
settings, harvest, adopt) against a *second* real node is design §11,
deferred until one exists to build against.

### `nodes.json` / `node_credentials.json`

Topology and credentials are **separate files**, deliberately: `nodes.json`
is safely readable and backupable (`id`, `label`, `agent_kind`, `address`,
`serving_address`, `added_ts` — no secrets); `node_credentials.json` is a
mode-`0600` sidecar (`{node_id: bearer_key}`), written with the same
atomic-tmp-then-chmod-then-rename discipline the rest of the deck's stores
use, and its values never leave `app/node_store.py` except through
`credential_for()`.

```json
// nodes.json
[
  {"id": "local",  "label": "autarch", "agent_kind": "local"},
  {"id": "sparky", "label": "sparky",  "agent_kind": "node-agent",
   "address": "http://192.168.1.x:7720",
   "serving_address": "http://192.168.1.x:8000", "added_ts": "..."}
]
```

`id` is **immutable identity** — it keys lifecycle intent (`<node>/<resource>`),
settings scopes (`<node>/<engine>`), and provenance artifact ids
(`oci:<node>:...`). There is no rename-id operation; `label` is the only
editable display string, and it must never build a key (the
`app/arbiter.py:1025-1050` rule, now structural: labels exist only in
registry entries). `id` is validated on create against a lowercase-slug
regex, refused (422) rather than coerced when it doesn't match; a duplicate
id is a 409. The `local` entry is seeded once, undeletable (409 on
`DELETE /api/nodes/local`), and is the only entry `agent_kind: "local"` may
ever describe — `add()` refuses that kind for anything else.

### Seed-once, and how to re-seed

At startup, `seed_if_missing()` runs **only while `nodes.json` does not
exist**: it seeds `local` (label from `MODEL_DECK_NODE_LABEL`), and, if
`MODEL_DECK_SPARK_NODE_URL` + `MODEL_DECK_SPARK_SERVING_URL` are both set,
seeds sparky with its address/serving-address and copies its credential out
of `ODS_REMOTE_NODE_KEYS[MODEL_DECK_SPARK_NODE_NAME]`. Once `nodes.json`
exists, **env is never consulted again** — no per-boot merge that could
resurrect a deleted entry. Compose keeps passing the env vars for exactly
one reason: a fresh install seeds itself on first boot.

The env-seeded sparky entry's id **must be exactly `LEGACY_SPARK_SEED_ID`**
(`app/node_store.py`, frozen migration data, literally `"sparky"`) — every
pre-N1 install's keyed data (intent, settings scopes, `oci:sparky:*`
provenance) already attaches through that exact string, so `seed_if_missing`
must keep reusing it verbatim rather than deriving it from anything. Get it
wrong and those records silently orphan rather than erroring;
`livetests/test_safe_nodes.py::test_sparky_seed_preserved_the_key_vocabulary`
is the live proof that they didn't. A **new** swap node — one added through
`POST /api/nodes` with `control: "swap"`, N1's multi-node generalization —
carries no such constraint: every keyed vocabulary (intent, settings scopes,
provenance, harvest routes) now derives from whatever registry id the
operator gives it, never a hardcoded literal.

**To re-seed:** delete `nodes.json` and restart. This re-reads the env vars
as if from a fresh install — **any operator edits made through the UI
(labels, addresses) are lost**, because the seed has no memory of what it
last wrote. `node_credentials.json` is untouched by this: a manually-set
credential in the sidecar survives unless the env var supplies a new one for
that same id, which overwrites it. To fully reset both topology and
credentials, delete both files.

### The credential is write-only

`credential` is accepted on `POST /api/nodes` and `PUT /api/nodes/{id}`,
surfaced only as `credential_set: true|false`, and **never appears in any
response, error body, or event detail.** `node-updated`'s event carries the
field *name* (`"credential"`) when one was supplied, never its value — other
fields (e.g. `label`) may still appear by value elsewhere (`node-added` logs
`label`), so this guarantee is specifically about the credential.

Credential rotation respects this too. `app.node_clients.NodeClients`
compares each swap node's live binding (address, serving_address, and the
credential's **sha256 fingerprint** via `NodeStore.credential_fingerprint`,
never a value) against what it last built a client from, on *every*
`client_for()` call — a changed fingerprint rebuilds the client (closing the
old one, never leaking it) exactly like a changed address or
serving_address. A rotated credential therefore reaches actuation on the
very next call, no restart involved, and the value itself never has to leave
the store to prove it changed. An unset credential fingerprints to `None`,
distinct from any real digest.

This closes a path that would otherwise leak it: pydantic v2's "missing
required field" errors carry the **entire parent object** as `input`, so a
422 for "address omitted" would echo the caller's credential back verbatim
in the response body under FastAPI's default handler. `app/main.py`'s
`RequestValidationError` handler strips the `input` key from *every* error
of *every* route (not just ones whose `loc` mentions "credential") before
serializing — closing the whole class for every request body shape, present
and future, rather than a name-coupled redaction that would silently miss
the next secret field some other router adds.

### Observer thread + status vocabulary

`app/node_observer.py::NodeObserver` runs on its **own daemon thread** (the
`app/update_check.py` precedent), never inside `arbiter.Watcher.tick()`:
that tick is one synchronous 2 s loop running the reconciler, and N nodes ×
a 5 s transport timeout on a down box would stall the machinery that keeps
local models loaded. It holds no intent store, no docker client, and a
client with no actuating verbs (`NodeAgentClient` is observe-only) — it
structurally cannot become a second actuator.

Every pass (`interval`, default 10 s) **re-reads the registry** (the
`dashboard-api/remote_nodes.py` 5 s re-read precedent), so add/remove/
credential edits made through the API apply to observation live, no
restart. Only `agent_kind: "node-agent"` entries are probed — `local` is
hardcoded `"online"` by `app/routers/status.py::_nodes_block` and never
asked. For each node-agent entry it probes `GET /v1/node/gpu` (governs
status) and `GET /v1/node/serving` (auxiliary — a failure there only
degrades `serving` to `null`, it never governs status itself) and writes a
snapshot atomically swapped in as one reference, so a reader never sees a
half-built pass.

| Status | Meaning |
|---|---|
| `online` | The gpu probe answered. A node that answers but reports its own collector failure still reads `online`, with the message carried in `error` |
| `offline` | Transport failure reaching the node (`NodeAgentUnreachable`) |
| `error` | The node answered badly — non-2xx or a bad body |
| `unconfigured` | No stored credential. **Never probed** — distinct from `offline` on purpose: "not set up" and "not answering" must not collapse into the same dot |

(`null` in `/api/state`'s `nodes[].status` is a fifth, implicit case: the
observer hasn't ticked this node yet — a fresh add, or a `NO_WATCHER`
deployment.)

### Sparky observed twice, on purpose

The existing lifecycle path (`SparkObserver` → `observe_spark` →
`derive_status`, TTL-cached, feeding the reconciler and the board's spark
card) is **untouched** and keeps answering "is the serving slot healthy".
`NodeObserver` answers a different question — "is the box answering at
all" — for the Nodes screen's status dots. Two consumers, two questions:
v1 does not route the reconciler, or the board's spark card, through the
new observation path. This is a named seam, not an oversight — a future
increment that tries to unify them needs to keep both questions answerable
separately.

### Live actuation rebinding (was: the restart-for-actuation caveat)

v1 shipped a caveat here: editing a node's address or credential applied to
**observation immediately** (`NodeObserver` re-reads the registry every
pass) but to **actuation only on the deck's next restart**, because
`SparkClient` was built once, at app startup, from whatever the registry
held at that moment. **N1 closed this** (`app/node_clients.py`): every
actuation path now takes its client from `NodeClients.client_for(node_id)`,
which compares the node's live registry binding against what it last built
a client from on *every* call and rebinds (closing the old client, never
leaking it) the moment address, serving_address, or the credential change —
no restart, and no per-node event needed to name the requirement, because
the requirement no longer exists.

### Deferred (design §11)

In intended order, from
`~/notes/designs/2026-08-09-model-deck-nodes-registry-design.md`:

1. ✅ **DONE (N1).** ~~Full N-node operability~~ — every single-spark
   hardcoded site is generalized off the registry's declared `control:
   "swap"` set: `SPARK_SLOT_KEY` is deleted (`slot_key(node_id)` derives the
   resource key per node), actuation goes through `app.node_clients`' one
   client per swap node instead of the one `SparkClient`, the harvest route
   builds one `(node_id, "vllm")` pair per swap node, the adopt allowlist
   (`POST /api/settings/adopt/{node}/{engine}`) accepts any control:"swap"
   node, and the spark router resolves whichever swap nodes are declared
   (`app.routers.serving.single_swap_node_id`, 409 naming candidates when
   more than one exists).
2. **Capability descriptors** — the ontology's per-node engine-capability
   schema. The node-agent's `/v1/node/info` `capabilities[]` is the
   designed hook; C2 shipped inference-from-harvested-catalog instead.
3. **Local-like nodes** — a second node exposing its own
   lemonade/comfyui/hipfire verbs needs `ui/src/model/nodes.ts`'s
   `controls` reworked first (see that file's header comment on the
   `App.tsx` local-snapshot prop-drilling landmine, dormant while only
   `node-agent` kinds can be added).
4. ✅ **DONE (N1).** ~~Live client rebinding~~ — actuation now picks up
   connection edits live, no restart (see above).
5. **Dashboard-api convergence** — shared node topology; today both
   dashboard-api and the deck read `ODS_REMOTE_NODE_KEYS` independently,
   and that stays.

## Provenance: where every artifact came from

The deck records the upstream origin and current version of every artifact it
can see — engine images, model weights, source repos, ComfyUI node packs — so
that "what was ds4 running last Tuesday, and where did it come from" has an
answer. This is the **ledger only** — provenance records, it never actuates.
Update-checking (below) is the first spec built on top of it: it reads the
declared `watch` list off each entry and writes back a verdict. Recipes and
backup export remain later specs.

### Three kinds, not four

Four artifact classes collapse to three origin kinds, because a ComfyUI node
pack *is* a git checkout — pinning KJNodes is a git-ref problem, not a fourth
mechanism. `role` (`engine` / `weights` / `source` / `nodepack` / `other`)
keeps the four-way distinction as data without a fourth code path.

| Kind | `artifact_id` | Version identity |
|---|---|---|
| `oci` | `oci:<node>:<repository>` | image digest |
| `git` | `git:<node>:<path>` | resolved commit |
| `file` | `file:<node>:<relpath>` | sha256, or `null` |

The tag is **not** part of the id: `ds4-spark` is one artifact whose version
moved v0.5.3 → v0.5.6. Weights key on `relpath`, never the catalog unit id —
`catalog.record_moved()` rewrites that id, so keying on it would orphan a
model's provenance exactly when the mover runs.

### `version` vs `label`

`current.version` is the exact machine identity nobody types (a digest, a
commit, a sha256). `current.label` is the string a human recognises
(`v0.5.6`). One is for comparing, the other for reading; neither is derivable
from the other, and `label: null` is honest for the many artifacts with no
human version. A single container inspect yields both — top-level `Image` and
`Config.Image`.

### Verification states

Four are stored; `stale` is computed at read time from `verified_at`, the way
`locations.describe()` computes `available`.

| State | Meaning |
|---|---|
| `exact` | Machine identity read directly this pass. |
| `consistent` | The cheap check passed but is not proof — size+mtime match, sha256 unknown. This is the honest state for weights. |
| `unavailable` | The source could not be reached. Deliberately distinct from `unknown`. |
| `unknown` | Never observed. Every `git` artifact is here in v1. |
| `stale` | Not stored. Reported when `verified_at` is older than `provenance_stale_s` (default 3600 s). `unavailable` never decays to `stale` — "the node is down" is the actionable fact. |

Only the on-demand deep check (`POST /api/provenance/verify`, which hashes the
file) can grade weights `exact`. A routine pass compares size and mtime, which
is a fingerprint, not a version.

⚠ **Read `verification` from the top level of a `GET /api/provenance` entry,
never from inside `current`.** The read layer removes the nested copy on the
way out precisely so there is one field with one answer — a consumer reading
the stored value would never see `stale`. The deep check's response reports
`matched_recorded` (`true`/`false`/`null`), which says whether the bytes
changed since the last hash; the entry is `exact` either way, because the file
was just hashed.

### Nothing converges

Provenance records a desired version and reports drift. **No code acts on
it**, and that is not a missing feature — it is currently outside the deck's
permissions:

- **autarch** — converging would need `docker pull` plus a container *create*.
  The socket proxy allows `start`/`stop`/`exec` only, and `exec` is pinned to
  one container precisely because a wildcard there would make the in-process
  guard the last thing between a deck bug and host-wide RCE.
- **sparky** — converging would need to edit an `image:` line in
  `compose-<profile>.yaml`. The node-agent serves compose **read-only** and
  has no docker access at all by design.

Widening either is a security decision, not an implementation detail. Drift
surfaces the way facts drift already does: reported, for a human to act on.

### Collection

- **Local images** — one `GET /containers/{name}/json` per park-allowlist
  container. Iterating known names rather than listing containers is what
  keeps this free of a new socket-proxy rule and a compose change.
- **Local weights** — from the catalog the deck already scans. No new I/O.
- **sparky images** — the `image:` line from the compose text the node-agent
  serves (declared), plus the digest from its harvested catalog (derived).
  ⚠ The digest is attributed **only** when `catalog["profile"]` names that
  exact profile. A node-agent older than the profile stamp, or a catalog
  belonging to a different profile, yields `version: null` — absent, never
  guessed. Sparky digests therefore require a node-agent carrying the
  `read_newest_catalog` profile stamp.
- **git** — declared only. The deck container cannot see host repos and there
  is no `git rev-parse` endpoint; `grade()` takes an injected `run_git` so the
  seam is named, and production passes `None`.
- **Declared** — `PUT /api/provenance/origin`. This is how aeon-vllm's
  "no source repo available" and its `/mnt/cold/images/` archive get recorded.
- **Backfill** — `POST /api/provenance/backfill` creates entries with
  `origin: null` for everything visible. It never fills in an origin; its
  output *is* the gap list.

### Files and endpoints

State lives in `/data/provenance.json` (atomic writes) and
`/data/provenance-history.jsonl` (append-only, fsync'd, one line per
transition, `to` embedding the whole block so a restore is one scan).

A corrupt `provenance.json` is **renamed aside** to
`provenance.json.corrupt-<ts>` rather than self-healed to empty — unlike
policy/registry/catalog, this file is the only home of operator-declared
origins.

```
GET    /api/provenance                      # ledger + gap list
GET    /api/provenance/history?artifact_id= # transitions, oldest first
POST   /api/provenance/backfill             # idempotent; never guesses an origin
PUT    /api/provenance/origin               # declare where something came from
PUT    /api/provenance/desired              # record what it SHOULD be (data only)
DELETE /api/provenance/desired?artifact_id= # "no opinion"
POST   /api/provenance/verify               # deep sha256 (file artifacts only)
DELETE /api/provenance?artifact_id=         # remove an entry (history survives)
PUT    /api/provenance/watch                # declare an artifact's upstream(s) to check
POST   /api/provenance/check                # run the update-check pass now (see below)
```

`artifact_id` is never a URL path segment — it contains `:` and can contain
`/` and `~`, so it travels as a query parameter or a body field.

`GET /api/state` carries a summary block:
`{"provenance": {"drift": [...], "gaps": <int>, "updates": <int>}}`.
`updates` is the count of artifacts whose latest check rolled up to
`available` — see [Update checking](#update-checking) below.

## Update checking

Provenance (above) answers "what is here now and where did it come from."
Update checking answers the next question — "is there something newer" —
without ever fetching, pulling, or building it. It reads whatever upstream an
operator **declared** for an artifact, compares it to the pinned value already
on file, and writes the verdict back onto that artifact's `update` field. It
never actuates: `app.reconcile` remains the only thing that changes what is
actually running, exactly as [Nothing converges](#nothing-converges)
already establishes for the rest of provenance.

### Four status words, and two of them are not synonyms

| Status | Meaning |
|---|---|
| `current` | Checked; the pinned value is the newest the declared order can see. |
| `available` | Checked; something ranked newer than the pin exists. |
| `undetermined` | **Reached** the upstream, but its answer cannot be ranked **and something other than the pin is out there** — an `order: "none"` tag set, or a pin that itself doesn't parse under the declared order. An unrankable set whose only tag *is* the pin is `current`, not `undetermined`: nothing unexplained was seen (`app/updates/oci.py:145`, `app/updates/git.py:184`). |
| `unavailable` | Could **not** reach the upstream, or could not even ask (rate-limited, network error, malformed response, a remote that has moved). |

`undetermined` and `unavailable` get confused because both mean "no verdict,"
but they point an operator in opposite directions. `undetermined` says *the
read succeeded — go look at the tag list yourself*, e.g. comfyui-aeon-spark's
`slim`/`full`/`latest` or llama.cpp's `b8763`-style build tags, neither of
which has an order this code is willing to invent. `unavailable` says *the
read itself failed* — a rate limit, a timeout, a moved repository — and the
last known-good verdict is retained rather than overwritten, the same
retention-over-erasure rule the rest of provenance already applies to a
locations/weights read that can't be reached.

A rollup (the `status` shown on an artifact with more than one watch source,
e.g. `ods-lemonade-server` below) is the **worst** of its sources
(`unavailable` > `undetermined` > `available` > `current`), so one healthy
source can never mask a sibling that is failing or unrankable.

### Ranking is declared per source, never inferred

Two of the four check types never rank anything — `git_compare` walks an
exact ahead/behind count against a pinned commit, and `oci_channel` compares
the digest a moving tag (`:slim`, `:latest`) currently resolves to against the
pinned digest. Neither needs an opinion about "newer."

The other two kinds, tag **sets** (`oci_tags` and `git_tags` — the
`_TAG_CHECKS` pair in `app/updates/__init__.py`), are where "newer" stops
being exact and becomes a convention — and the convention is written down per
source, never guessed from the tag strings themselves:

| Check | Needs | `order` | Ranks by |
|---|---|---|---|
| `git_compare` | `remote`, `ref`, `pinned` | must be `null` | ahead/behind count (exact) |
| `oci_channel` | `registry`\*, `repository`, `reference`, `pinned` | must be `null` | digest equality (exact) |
| `git_tags` | `remote`, `pinned`, `order` | `"semver"` \| `"date"` \| `"none"` | parsed tag comparison |
| `oci_tags` | `registry`\*, `repository`, `pinned`, `order` | `"semver"` \| `"date"` \| `"none"` | parsed tag comparison |

\* `registry` defaults to `ghcr.io` when omitted.

Every column entry above except the starred one is **enforced** by
`app.updates.validate_watch`, as a non-empty string: a source whose checker
could not execute it is refused at the door (422, nothing written) rather
than accepted and then reported permanently `unavailable`.

`order: "none"` is not "not yet configured" — it is the **honest** choice for
a tag set with no sane order at all (llama.cpp's `b8763` build tags; a channel
name is handled by `oci_channel` instead and never reaches ranking). A tag
that doesn't parse under the declared order is excluded from ranking and
listed in `detail.unranked`, never coerced into a comparison — the one place
in this whole package that could be *wrong* rather than merely unavailable,
so it refuses to guess.

### Its own thread, never the watcher tick

The watcher tick that drives arbitration/reconciliation runs every ~2 s and
must stay fast — a stalled tick is a stalled reconciler, which is what keeps
models loaded. `UpdateChecker` runs on its **own** daemon thread instead, at
`update_interval_s` (default 6 h — upstream releases land on the order of
weeks, and it shares GitHub's 60 requests/hour anonymous ceiling with
everything else on the box). `POST /api/provenance/check` runs the identical
pass synchronously, on the request thread, for "check now" — same code,
same per-source/per-artifact failure isolation, just not on a timer.

A pass **reads** upstreams and **writes** the provenance ledger only. It never
imports `app.reconcile` or `app.intent`, so it cannot become a second
actuator even by accident — the same class of guarantee [Nothing
converges](#nothing-converges) states for the rest of
provenance, here true structurally rather than merely by convention.

### Kill switch

`MODEL_DECK_UPDATE_CHECK_ENABLED=false` (default `true`) stops the background
thread from ever starting, and makes `POST /api/provenance/check` a harmless
no-op (`{"checked": 0, "available": 0}`, no upstream contacted) instead of
running a pass. This is the only part of the deck that talks to the public
internet, so turning it off is one flag, not a rebuild or a network policy
change: set it in `ods/.env` and recreate the container — `compose.yaml`
passes it (and `MODEL_DECK_UPDATE_INTERVAL_S`) through, and both are declared
in `ods/.env.schema.json`. (`POST /api/provenance/check` answering `503` is a different case —
no `UpdateChecker` was constructed at all, e.g. `MODEL_DECK_NO_WATCHER=1`.)

One consequence to know before flipping it. The collector derives a `channel`
watch source from any digest-pinned `origin` on its own, and the only thing
that distinguishes "the operator cleared this deliberately" from "nobody has
looked at it yet" is whether the artifact has ever been checked — i.e.
whether `entry.update` exists, which **only a check pass writes**. With
checking disabled nothing ever writes it, so a watch cleared with
`PUT /watch {"sources": []}` is re-derived on the *next* collector pass, and
every pass after that. To make a clear stick while checking is off, withdraw
the origin instead (`PUT /api/provenance/origin` with `"origin": null`) —
with nothing to derive from, nothing comes back.

### `PUT /api/provenance/watch`

Declares (replaces, whole-list) the upstream sources checked for one artifact:

```json
PUT /api/provenance/watch
{
  "artifact_id": "oci:local:ods-lemonade-server",
  "sources": [
    {"id": "sdk", "check": "oci_tags", "registry": "ghcr.io",
     "repository": "lemonade-sdk/lemonade-server",
     "pinned": "v10.2.0", "order": "semver"},
    {"id": "llama-cpp", "check": "git_tags",
     "remote": "https://github.com/ggml-org/llama.cpp",
     "pinned": "b8763", "order": "none"}
  ]
}
```

Every source needs a non-empty **string** `id`, unique within the artifact —
it is the key `record_update` merges results on, so a duplicate is refused
(422) rather than letting one source's verdict silently suppress the other's.
It also needs a `check` from the table above, plus every field that table
lists for that check (including `pinned`), each a non-empty string.
`sources: []` clears the watch —
the artifact stops being checked, and any prior verdict for a dropped source
id is dropped with it (retention is bounded by what is still watched). A
malformed source is rejected **whole** (422, nothing written) rather than
partially applied — see `app.updates.validate_watch`.

## Safety invariants (12–17): storage

The storage feature enforces six safety invariants (continuing the deck's numbered safety list):

**12.** Source is never deleted before the destination is hash-verified (sha256 computed in-flight, re-read after fsync). No reachable zero-copies state.

**13.** A loaded/serving model and the litellm default-route model are never moved — the latter with no force override.

**14.** Unavailable ≠ empty: catalog entries survive an unplugged drive, and the deck never writes into an unavailable location (marker-file check).

**15.** One move at a time; every phase crash-safe; startup janitor removes orphans.

**16.** Auto-eviction respects pins, only targets the location's declared `archive_to`, and never touches hipfire's store (out of scope v1).

**17.** Every storage mutation — manual or watcher — is audited to `events.jsonl`.

## Architecture

```
Model Deck UI (:3015)
       │
       ▼
Model Deck API (:3015, FastAPI)
  ├── app/routers/
  │    ├── control.py ──────────── tenant load/unload/free/park
  │    ├── storage.py ──────────── locations, moves, pins, policy
  │    ├── policy.py ──────────── VRAM policy, autoheal
  │    ├── sets.py ────────────── Set Builder (load groups)
  │    ├── serving.py ─────────── Per-node swap control (/api/nodes/{id}/serving/*)
  │    ├── lifecycle.py ──────── Quarantine release + adoption
  │    └── status.py ──────────── System, tenant, and lifecycle status
  ├── app/
  │    ├── arbiter.py ────────────── VRAM policy enforcement (watcher + pure planner)
  │    │                             + the reconcile pass (end of every tick)
  │    ├── intent.py ──────────── IntentStore — durable desired state (intent.json)
  │    ├── lifecycle.py ──────── derive_status: intent × observation → one status
  │    ├── observe.py ─────────── Engine vocabularies → one observation shape
  │    ├── reconcile.py ───────── plan_reconcile: which statuses justify acting
  │    ├── storage.py ──────────── Storage tiering (watcher + pure planner)
  │    ├── locations.py ────────── Location store, marker files
  │    ├── catalog.py ─────────── Model catalog (scanner, unit tracking)
  │    ├── mover.py ──────────────── Efficient cross-drive copy (hash-verify, atomic)
  │    ├── policy.py ──────────── VRAM policy store
  │    ├── sets.py ────────────── Set registry
  │    ├── registry.py ────────── Live model scan (lemonade + ComfyUI)
  │    ├── events.py ────────────── Audit trail (events.jsonl)
  │    ├── engine_kinds.py ────── Declared-engine kind schemas + adapters
  │    │                             (spec §8: the ONE module allowed to
  │    │                             know a kind name outside its own
  │    │                             engines/ client — see Declared Engines)
  │    ├── local_clients.py ───── Per-resource client resolution, live off
  │    │                             node_store's local `engines[]`
  │    └── engines/
  │         ├── lemonade.py ───── llama.cpp client
  │         ├── comfyui.py ────── ComfyUI client
  │         ├── hipfire.py ────── hipfire client
  │         ├── sglang_omni.py ── remote engine client, over node_agent.py's
  │         │                     up/down/status channel (fourth kind)
  │         └── docker_ctl.py ─── container restart/stop (for engine notify)
  ├── ui/ ────────────────────────── React frontend
  │    └── dist/ ────────────────── Built assets (served at /)
  └── tests/ ─────────────────────── Pytest suite (pure + integration)
```

## Files

- `app/main.py` — FastAPI application, startup, exception handlers
- `app/routers/` — Endpoint modules (control, storage, policy, sets, spark, serving, lifecycle, status, nodes)
- `app/node_store.py` — NodeStore: `nodes.json` topology + `node_credentials.json` 0600 sidecar, seed-once migration
- `app/node_observer.py` — NodeObserver: own daemon thread, registry-driven status probes (see [Node registry](#node-registry-topology-credentials-and-observation))
- `app/engines/node_agent.py` — Thin observe-only client (`info`/`gpu`/`serving`); `SparkClient` extends the same base
- `app/arbiter.py` — VRAM arbitration, watcher, planning, lifecycle reconcile pass
- `app/intent.py` — IntentStore: durable desired state (`/data/intent.json`), failure budget
- `app/lifecycle.py` — `derive_status`: intent × observation → one status (pure)
- `app/observe.py` — Per-engine observation adapter + `SparkObserver` (TTL cache, backoff)
- `app/reconcile.py` — `plan_reconcile`: the one status (`down`) that justifies acting (pure)
- `app/storage.py` — Storage tiering logic, watermark rules, StorageWatcher
- `app/locations.py` — Location store, marker files, availability checks
- `app/catalog.py` — Model catalog scanner, unit state
- `app/mover.py` — Cross-drive file mover (copy, hash-verify, atomic rename)
- `app/policy.py` — VRAM policy store and validation
- `app/engine_kinds.py` — Declared-engine kind schemas, adapters (observe/actuate/verbs) — see [Declared Engines](#declared-engines)
- `app/local_clients.py` — Resolves each declared resource to its client, live off `node_store`
- `app/sets.py` — Set Builder registry and executor
- `app/registry.py` — Live model scanner for lemonade and ComfyUI
- `app/notify.py` — Engine notification hooks (lemonade restart, ComfyUI refresh)
- `app/engines/` — Engine client libraries
- `app/events.py` — Audit event logging
- `app/settings.py` — Configuration, environment variables
- `ui/` — React source code
- `ui/gates/` — `deck-gate`'s browser-gate harness (see [Testing](#testing) below)
- `tests/` — Pytest test suite
- `livetests/` — `deck-drill`'s live capability suite (see [Testing](#testing) below)
- `Dockerfile` — Multi-stage container (Node + Python)
- `requirements.txt` — Python dependencies

## Testing

Two live/scripted suites sit beside the unit suites (`pytest`, `npm test`), each with its
own runner script at the extension root and its own detailed README. **Do not confuse
them** — they exercise different halves of the stack:

- **`./deck-drill`** exercises the **backend** against a live box — real API calls, safe
  tier reversible-by-finalizer, disruptive tier gated behind a pre-flight window. Details:
  `livetests/README.md`.
- **`./deck-gate`** exercises the **UI** — headless Chrome (`playwright-core`, driving the
  system `google-chrome`, never a downloaded browser binary) clicking through the real
  built bundle. Fixture tier (default) is deterministic and needs nothing running; `--live`
  is a strictly read-only fidelity check against a live deck. Full details, the two tiers,
  the stub-never-derives rule, and known limitations: `ui/gates/README.md`.

Both share the same shape deliberately (exit codes `0`/`1`/`2`, `-k` selection, a
`~/notes/evidence/deck-<name>s/` report directory) — that convergence is intentional.
Shared *code* between them is not: `deck-drill` has no idea what a browser is, and
`deck-gate` never **writes** to a backend route outside its stub (`--live`'s fidelity check
does call the live deck directly, but GET-only — see `ui/gates/README.md`'s "two tiers"
section) and never exercises backend *behaviour* at all, written or read: the add/forget
flows it clicks through are proven only against the stub's scripted responses, never against
a real box actually accepting or persisting anything.

`playwright-core` is a **devDependency of `ui/` only** and must never resolve inside the
built container — verify with:

```bash
docker build -t model-deck-packaging-probe .
docker run --rm model-deck-packaging-probe sh -c \
  'ls /srv/app/ui/gates 2>&1; node -e "require(\"playwright-core\")" 2>&1 | head -1'
```

Expect `ui/gates` absent from the image (check `/srv/app`, not `/app` — the image's
`WORKDIR` is `/srv`, not `/app`) and `playwright-core` unresolvable.

## Troubleshooting

**Model not appearing in Load dropdown:**
- Check the registry scan: `GET /api/state` → `models` lists what was found in lemonade's store
- If missing, verify the compose mount and lemonade's `MODEL_PATH` environment variable

**Pull-through load fails with "lemonade not ready":**
- Lemonade is restarting to register the pulled file. The deck waits up to 60 seconds.
- Check lemonade logs: `docker compose logs ods-llama-server`
- If it doesn't come back, restart manually: `docker compose restart ods-llama-server`

**An engine died and the deck did not restore it:**
- `GET /api/state` → `lifecycle["local/<engine>"]`. `parked` means the deck has a recorded *deliberate unload* — resume it normally (that also re-records intent). `unmanaged` means no intent was ever recorded, so there is nothing to restore to: adopt it (`POST /api/lifecycle/adopt/local/<engine>`) while it is healthy, or just use the normal load/park/resume routes, which record intent themselves.
- `unreachable` means the deck could not reach the engine at all — that is a probe/network problem, not a dead model, and it is deliberately not actionable.
- `quarantined` means two restores in a row raised. Fix the cause, then `POST /api/lifecycle/quarantine/local/<engine>/clear`. Check `events.jsonl` for `lifecycle-restore-failed`.
- Also confirm automation is on: `policy.json` must not contain `{"_auto": {"enabled": false}}` (it defaults to enabled).

**Something the deck parked keeps coming back:**
- That would be a bug in the reconciler, which never acts on `parked` (drill D9). Far more likely it is the *arbiter*'s pending-load healing reloading the litellm default-route model — a different mechanism entirely (see HealSuppressor / VRAM policy), and it logs `load-retriggered`, not `lifecycle-restore`.

**"Location unavailable — drive unmounted?" when moving:**
- The bind mount in compose.yaml is missing or the drive is disconnected
- Confirm: `docker compose exec ods-model-deck ls /stores/<name>` should list the mount
- Verify the `.deck-store.json` marker file is present: `ls /stores/<name>/.deck-store.json`
- If the drive is genuinely gone, deregister the location (`DELETE /api/storage/locations/{name}`) and re-register after remounting

**Watermark eviction not kicking in:**
- Check the auto mode: `GET /api/storage/policy` should show `"auto": true`
- Verify the location has a `watermark_gb` set (not null) and a valid `archive_to` destination
- Run `POST /api/storage/rescan` to force a catalog update
- Check `events.jsonl` for `storage_suggestion` or `storage_shortfall` events (watcher runs every 60 s)

**Models appear "unavailable" in the catalog:**
- A location's marker file is missing or does not match the stored UUID
- Confirm the mount is active: `docker compose exec ods-model-deck ls /stores/<name>`
- **If the marker file exists on the physical media** (normal case after remount), remounting the drive alone restores availability — no further action needed; the deck will see the marker on the next scan
- **If the marker file is truly lost** (e.g., drive corruption), deregister the location (`DELETE /api/storage/locations/{name}`) and re-register it. **Warning:** deregistering removes the location's catalog entries, losing pin state and `last_used` usage tracking for those models. Re-register only if the marker cannot be restored

## License

Part of ODS — Local AI Infrastructure
