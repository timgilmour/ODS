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
    model_config = SettingsConfigDict(env_prefix="MODEL_DECK_", extra="ignore")

    # --- Storage ---
    data_dir: str = "/data"

    # --- Lemonade ---
    lemonade_url: str = "http://llama-server:8080"
    lemonade_key: str = Field(default="", validation_alias="LEMONADE_API_KEY")
    # The wrapped llama-server's own port (Lemonade passes --host 0.0.0.0
    # --metrics through); the wrapper's /metrics serves its web UI instead.
    lemonade_metrics_url: str = "http://llama-server:8001/metrics"

    # --- ComfyUI ---
    comfyui_url: str = "http://comfyui:8188"

    # --- LiteLLM ---
    litellm_url: str = "http://litellm:4000"
    litellm_key: str = Field(default="", validation_alias="LITELLM_KEY")

    # --- Spark (remote single-slot serving node; lifecycle only) ---
    # Empty node/serving URL leaves the spark engine unbuilt (the /api/spark
    # endpoints answer 503). The bearer key comes from the stack-wide
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

    # --- Docker control (tecnativa/docker-socket-proxy sidecar) ---
    dockerctl_url: str = "http://docker-ctl:2375"

    # --- Parking / arbitration ---
    hipfire_container: str = "ods-hipfire"
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

    # Seconds a deliberate lemonade unload (manual, set-apply, or the
    # watcher's own idle release) suppresses contention healing's pending-load
    # inference, so healing can't immediately revert it. A subsequent load
    # clears it early. See app.arbiter.HealSuppressor.
    heal_suppress_s: int = 600

    # GPU list indices in read_gpus' filtered order (0-based over qualifying
    # cards). On this box hipfire owns index 0; lemonade + comfyui share
    # index 1 (the arbiter only ever contends for VRAM on the lemonade GPU).
    lemonade_gpu_index: int = 1
    hipfire_gpu_index: int = 0

    # Container bind for the sysfs GPU reader (see compose.yaml volumes).
    # /sys is mounted whole at /sysfs so the /sys/class/drm symlinks into
    # /sys/devices resolve inside the container.
    drm_root: Path = Path("/sysfs/class/drm")
    kfd_root: Path = Path("/sysfs/class/kfd/kfd/proc")
