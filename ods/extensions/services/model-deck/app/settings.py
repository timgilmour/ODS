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
environment — ``admin_token`` defaults to the empty string, which disables
mutating endpoints for tests and bare-uvicorn runs only; real deployments
always set ``MODEL_DECK_ADMIN_TOKEN`` because compose.yaml's ``:?`` guard
refuses to start the container without a non-empty value.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_DECK_", extra="ignore")

    # --- Storage ---
    data_dir: str = "/data"

    # --- Auth ---
    # Empty string disables mutating endpoints (for tests and bare-uvicorn runs
    # only); real deployments always set MODEL_DECK_ADMIN_TOKEN because
    # compose.yaml's :? guard refuses to start the container without it.
    admin_token: str = ""

    # Shared secret the reverse proxy (ods-lan Caddy) injects as
    # X-Deck-Proxy-Key alongside Authelia's Remote-Groups header, proving the
    # Remote-Groups header actually came through the proxy rather than being
    # forged by a sibling compose container. Empty string disables the
    # Remote-Groups auth branch entirely (see app.security.require_admin).
    proxy_key: str = ""

    # --- Lemonade ---
    lemonade_url: str = "http://host.docker.internal:13305"
    lemonade_key: str = Field(default="", validation_alias="LEMONADE_API_KEY")

    # --- ComfyUI ---
    comfyui_url: str = "http://comfyui:8188"

    # --- LiteLLM ---
    litellm_url: str = "http://litellm:4000"
    litellm_key: str = Field(default="", validation_alias="LITELLM_KEY")

    # --- Host agent ---
    hostagent_url: str = "http://host.docker.internal:7710"
    hostagent_key: str = Field(default="", validation_alias="HOST_AGENT_KEY")

    # --- Docker control (tecnativa/docker-socket-proxy sidecar) ---
    dockerctl_url: str = "http://docker-ctl:2375"

    # --- Parking / arbitration ---
    hipfire_container: str = "ods-hipfire"
    park_allowlist: list[str] = Field(
        default_factory=lambda: ["ods-hipfire", "ods-comfyui", "ods-llama-server"]
    )
    watch_interval: float = 2.0

    # GPU list indices in read_gpus' filtered order (0-based over qualifying
    # cards). On this box hipfire owns index 0; lemonade + comfyui share
    # index 1 (the arbiter only ever contends for VRAM on the lemonade GPU).
    lemonade_gpu_index: int = 1
    hipfire_gpu_index: int = 0

    # Container binds for the sysfs GPU reader (see compose.yaml volumes).
    drm_root: Path = Path("/sysdrm")
    kfd_root: Path = Path("/syskfd")
