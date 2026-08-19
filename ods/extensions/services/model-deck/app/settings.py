"""
Model Deck runtime configuration.

Most settings are read under the ``MODEL_DECK_`` env prefix (e.g.
``MODEL_DECK_ADMIN_TOKEN`` -> ``admin_token``). A handful of credentials are
shared with sibling ODS extensions and must be read under their own exact
env var names instead of the prefixed form — Lemonade, LiteLLM, and the
host-agent all mint/consume these same names elsewhere in the stack, so
aliasing here keeps a single source of truth per credential rather than
duplicating it under a Model Deck-specific name.

All fields default to a usable value so ``Settings()`` never requires an
environment. There is deliberately NO auth config: the admin-token/proxy-key
gate was removed 2026-07-22 (ops-first on a single-operator box; the LAN
path still sits behind Authelia via ods-lan).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_DECK_", extra="ignore", env_ignore_empty=True)

    # --- Storage ---
    data_dir: str = "/data"

    # --- Identity ---
    # What this box is called on the board. Deliberately NOT the container
    # hostname, which is a random hex string under compose. Set per install
    # via MODEL_DECK_NODE_LABEL; "local" is a truthful fallback, not a guess.
    node_label: str = "local"

    # --- Lemonade ---
    lemonade_url: str = "http://llama-server:8080"
    lemonade_key: str = Field(default="", validation_alias="LEMONADE_API_KEY")
    # The wrapped llama-server's own port (Lemonade passes --host 0.0.0.0
    # --metrics through); the wrapper's /metrics serves its web UI instead.
    lemonade_metrics_url: str = "http://llama-server:8001/metrics"

    # --- ComfyUI ---
    comfyui_url: str = "http://comfyui:8188"

    # --- Dashboard API (local GPU telemetry pass-through; ontology ruling:
    # telemetry is CONSUMED from dashboard-api, never rebuilt) ---
    dashboard_api_url: str = "http://dashboard-api:3002"
    dashboard_api_key: str = Field(default="", validation_alias="DASHBOARD_API_KEY")
    # Fallback for the STOCK install, where DASHBOARD_API_KEY is unset:
    # dashboard-api mints a random key into this file on first start, and
    # /api/gpu/detailed requires a bearer, so a deck that cannot read it 401s
    # forever. Same file the dashboard's own nginx entrypoint reads
    # (extensions/services/dashboard/entrypoint.sh:5-20), reached through the
    # ro ${ODS_DATA_DIR}:/ods-data mount in compose.yaml. Absent/unreadable ⇒
    # no header, exactly as before (app/telemetry.py's _auth_headers).
    dashboard_api_key_file: str = "/ods-data/dashboard-api-key.txt"

    # --- LiteLLM ---
    litellm_url: str = "http://litellm:4000"
    litellm_key: str = Field(default="", validation_alias="LITELLM_KEY")

    # --- Spark (remote single-slot serving node; lifecycle only) ---
    # One-time registry seeds (node_store.seed_if_missing, a no-op once
    # nodes.json exists): left empty on a fresh install, no sparky node is
    # seeded, so its serving routes 404 as an unknown node and rename's
    # single-swap-node resolver 503s. The bearer key comes from the stack-wide
    # ODS_REMOTE_NODE_KEYS JSON map (the same credential dashboard-api's
    # remote-node poller consumes), selected by spark_node_name.
    spark_node_url: str = ""
    spark_node_name: str = "sparky"
    spark_serving_url: str = ""
    spark_node_keys_json: str = Field(default="",
                                      validation_alias="ODS_REMOTE_NODE_KEYS")

    # --- Host agent ---
    # Empty = auto-derive from this container's default gateway. The host
    # agent binds the ods-network host-side gateway (e.g. 172.18.0.1), which
    # host.docker.internal does NOT reach (it maps to the default bridge,
    # blocked by Docker's inter-network isolation). Mirrors dashboard-api's
    # config.py:_detect_container_default_gateway.
    hostagent_url: str = ""
    hostagent_key: str = Field(default="", validation_alias="HOST_AGENT_KEY")

    # --- Docker control (wollomatic/socket-proxy sidecar) ---
    dockerctl_url: str = "http://docker-ctl:2375"

    # --- Parking / arbitration ---
    # hipfire_container feeds the one-time seed below and deck["hipfire"]
    # (kept for the pin tests); after ruling #4c, production never reads
    # deck["hipfire"] (all actuation routes through local_clients.client_for).
    hipfire_container: str = "ods-hipfire"
    # SEED ONLY since E1 (Task 9): fed `seed_engines_if_missing`'s
    # legacy-triple connection.container field (app/node_store.py:452) at
    # most once, on an upgrading box's first boot. The storage notify hook
    # that used to restart THIS specific container name now iterates the
    # LIVE declaration and reads each resource's OWN connection.container
    # instead (app/notify.py's own docstring) — this setting has no other
    # reader.
    lemonade_container: str = "ods-llama-server"
    # Seconds after hipfire's last observed served request during which
    # park/apply refuse to restart it (the single-slot conversation cache
    # means a restart between turns costs the next turn minutes of
    # re-prefill). 0 disables the recency rule; an in-flight request
    # (queue_depth > 0) always refuses. See HipfireClient.ensure_not_busy.
    hipfire_activity_window_s: float = 600.0
    park_allowlist: list[str] = Field(
        default_factory=lambda: ["ods-hipfire", "ods-comfyui", "ods-llama-server"]
    )
    watch_interval: float = 2.0

    # Node-observation cadence (app/node_observer.py). Its own thread, never
    # the watcher tick — N nodes x a 5 s transport timeout on a down box
    # would stall the reconciler. 10 s (dashboard-api's remote poller runs
    # at 5 s) is fresh enough for status dots.
    node_observe_interval_s: float = 10.0

    # Characteristics refresh cadence. The watcher ticks every ~2 s;
    # re-reading every checkpoint that often is pointless I/O, so derivation
    # is throttled to this interval — except the first successful lifecycle
    # restore of an incident clears the throttle (no new I/O in the restore
    # path itself; see Watcher._execute_restore), so the very next derive
    # pass captures live facts while they're freshest instead of waiting up
    # to this many seconds. That clear is itself floor-limited
    # (_DERIVE_RESTORE_FLOOR_S, 30 s) so a crash-looping resource — restores
    # succeeding every tick without ever raising, so the failure-budget/
    # quarantine machinery never trips — can't turn into a derive on every
    # tick for the whole incident.
    derive_interval_s: float = 300.0

    # Provenance refresh cadence. Images and repos change on the order of
    # weeks, not the ~2 s tick, so this pass is throttled far harder than
    # arbitration. Separate from derive_interval_s because the two answer
    # different questions ("what is this model" vs "where did it come from")
    # and should be tunable apart.
    provenance_interval_s: float = 300.0
    # How old a successful verification may be before the READ side reports
    # it as `stale` (app.provenance.describe). Never stored — computed at
    # read time, the way app.locations.describe computes `available`.
    provenance_stale_s: float = 3600.0

    # Update-checking cadence. Upstream releases land on the order of weeks,
    # and the anonymous GitHub ceiling (60 requests/hour per IP, shared with
    # everything else on this box) is the real constraint -- so this is
    # throttled far harder than any other pass. Runs on its own thread, never
    # on the watcher tick, because a network call there would stall the
    # reconciler.
    update_interval_s: float = 21600.0
    # Kill switch. This is the only part of the deck that talks to the public
    # internet; turning it off must be one flag, not a rebuild.
    update_check_enabled: bool = True

    # Seconds a deliberate lemonade unload (manual, set-apply, or the
    # watcher's own idle release) suppresses contention healing's pending-load
    # inference, so healing can't immediately revert it. A subsequent load
    # clears it early. See app.arbiter.HealSuppressor.
    heal_suppress_s: int = 600

    # SEED ONLY since E1 (Task 3/9): GPU list indices in read_gpus' filtered
    # order (0-based over qualifying cards) — fed ONLY
    # `seed_engines_if_missing`'s legacy-triple gpu_index fields
    # (app/node_store.py:447,453,457) at most once, on an upgrading box's
    # first boot. The arbiter has read per-resource placement from each
    # entry's OWN declared `gpu_index` since Task 3/5 (see
    # Watcher._infer_pending's own docstring: "kills the single global
    # settings.lemonade_gpu_index"); neither setting has any other reader.
    # On the seeded box these values still describe the real topology
    # (hipfire owns index 0; lemonade + comfyui share index 1) — that
    # placement now lives in nodes.json, not here.
    lemonade_gpu_index: int = 1
    hipfire_gpu_index: int = 0

    # Container bind for the sysfs GPU reader (see compose.yaml volumes).
    # /sys is mounted whole at /sysfs so the /sys/class/drm symlinks into
    # /sys/devices resolve inside the container.
    drm_root: Path = Path("/sysfs/class/drm")
    kfd_root: Path = Path("/sysfs/class/kfd/kfd/proc")

    # --- Storage tiering ---
    storage_watch_interval: float = 60.0
    # Free-space slack required at a move destination beyond the unit size.
    storage_slack_bytes: int = 2_000_000_000
