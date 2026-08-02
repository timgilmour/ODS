# model-deck

Model Deck is the centralized VRAM control plane for ODS, arbitrating GPU memory across multiple engines (lemonade, ComfyUI, hipfire) with a pluggable policy system. It also manages model storage tiers (hot/cold disk moves) and node lifecycle (Spark single-slot GPU scheduling).

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

Settings (Python `MODEL_DECK_*` env prefix, or defaults in `app/settings.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_DECK_STORAGE_WATCH_INTERVAL` | `60` | Watermark check cadence (seconds) |
| `MODEL_DECK_STORAGE_SLACK_BYTES` | `2e9` | Headroom to reserve when planning moves (2 GB) |
| `MODEL_DECK_LEMONADE_CONTAINER` | `ods-llama-server` | Docker container name for lemonade (needed to restart for model registration) |

## API Endpoints

### Core Status

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | Tenant status (GPU, loaded model, queue, default route) |
| `GET` | `/api/registry` | Scan lemonade and ComfyUI stores for available models |
| `GET` | `/api/policy` | Current VRAM policy and set registry |

### Tenant Control

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/tenants/lemonade/load` | Load a model in lemonade (supports `?pull=true` for cold models, `?force=true` to override host-agent guard) |
| `POST` | `/api/tenants/lemonade/unload` | Unload current model or a named model |
| `POST` | `/api/tenants/comfyui/free` | Free ComfyUI VRAM (unload all models) |
| `POST` | `/api/tenants/hipfire/park` | Park hipfire context (freeze inference, free VRAM) |
| `POST` | `/api/tenants/hipfire/resume` | Resume hipfire context |

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

### Spark (Remote Node)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/spark/nodes` | List known Spark nodes and GPU availability |
| `POST` | `/api/spark/swap` | Hot-swap a model in Spark (move ↔ load) |

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

The watcher ticks every 60 seconds, respects the `auto: on/off` toggle, and yields to in-flight set applies and move jobs.

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

## Safety invariants (12–17)

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
  │    ├── spark.py ──────────── Remote Spark node swap
  │    └── status.py ──────────── System and tenant status
  ├── app/
  │    ├── arbiter.py ────────────── VRAM policy enforcement (watcher + pure planner)
  │    ├── storage.py ──────────── Storage tiering (watcher + pure planner)
  │    ├── locations.py ────────── Location store, marker files
  │    ├── catalog.py ─────────── Model catalog (scanner, unit tracking)
  │    ├── mover.py ──────────────── Efficient cross-drive copy (hash-verify, atomic)
  │    ├── policy.py ──────────── VRAM policy store
  │    ├── sets.py ────────────── Set registry
  │    ├── registry.py ────────── Live model scan (lemonade + ComfyUI)
  │    ├── events.py ────────────── Audit trail (events.jsonl)
  │    └── engines/
  │         ├── lemonade.py ───── llama.cpp client
  │         ├── comfyui.py ────── ComfyUI client
  │         ├── hipfire.py ────── hipfire client
  │         └── docker_ctl.py ─── container restart/stop (for engine notify)
  ├── ui/ ────────────────────────── React frontend
  │    └── dist/ ────────────────── Built assets (served at /)
  └── tests/ ─────────────────────── Pytest suite (pure + integration)
```

## Files

- `app/main.py` — FastAPI application, startup, exception handlers
- `app/routers/` — Endpoint modules (control, storage, policy, sets, spark, status)
- `app/arbiter.py` — VRAM arbitration, watcher, planning
- `app/storage.py` — Storage tiering logic, watermark rules, StorageWatcher
- `app/locations.py` — Location store, marker files, availability checks
- `app/catalog.py` — Model catalog scanner, unit state
- `app/mover.py` — Cross-drive file mover (copy, hash-verify, atomic rename)
- `app/policy.py` — VRAM policy store and validation
- `app/sets.py` — Set Builder registry and executor
- `app/registry.py` — Live model scanner for lemonade and ComfyUI
- `app/notify.py` — Engine notification hooks (lemonade restart, ComfyUI refresh)
- `app/engines/` — Engine client libraries
- `app/events.py` — Audit event logging
- `app/settings.py` — Configuration, environment variables
- `ui/` — React source code
- `tests/` — Pytest test suite
- `Dockerfile` — Multi-stage container (Node + Python)
- `requirements.txt` — Python dependencies

## Troubleshooting

**Model not appearing in Load dropdown:**
- Check registry: `GET /api/registry` shows models found in lemonade's store
- If missing, verify the compose mount and lemonade's `MODEL_PATH` environment variable

**Pull-through load fails with "lemonade not ready":**
- Lemonade is restarting to register the pulled file. The deck waits up to 60 seconds.
- Check lemonade logs: `docker compose logs ods-llama-server`
- If it doesn't come back, restart manually: `docker compose restart ods-llama-server`

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
