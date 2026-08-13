"""Probe what this node is serving (OpenAI-compatible endpoint + container)."""
import logging
import subprocess

import httpx

import nodeconfig
import swapctl

_logger = logging.getLogger(__name__)

# The misconfiguration that blinded serving detection for 4 days (2026-08-12
# incident): with vLLM profiles configured and NODE_SERVING_PROBE_URL unset,
# the vllm/env probe path returns all-null forever, and the only signal was
# this container's own logs — where signals go to die. The probe result now
# carries this text so the deck can surface it (as the
# lifecycle-node-misconfigured event, model-deck app/arbiter.py
# _node_observations).
PROBE_URL_WARNING = ("vllm profiles configured but NODE_SERVING_PROBE_URL is "
                     "unset — serving detection is blind")


class ProbeError(RuntimeError):
    pass


def _fetch_raw(url: str) -> httpx.Response:
    """GET url, raising ProbeError unless the response is 2xx.

    Shared by `_fetch_models_payload` (which parses the body as OpenAI-shaped
    JSON) and `_probe_health_2xx` (which only cares about the status code).
    """
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        return resp
    except httpx.HTTPError as exc:
        raise ProbeError(str(exc)) from exc


def _fetch_models_payload(url: str) -> dict:
    try:
        return _fetch_raw(url).json()
    except ValueError as exc:
        raise ProbeError(str(exc)) from exc


def _probe_health_2xx(url: str) -> bool:
    """GET url; any 2xx response -> True. No OpenAI-shaped payload is parsed
    or expected -- this is a bare liveness check for non-vLLM engines."""
    try:
        _fetch_raw(url)
        return True
    except ProbeError:
        return False


def _container_status(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True, text=True, timeout=2.0,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _probe_env_configured() -> dict:
    """The original probe path: env-configured OpenAI /v1/models endpoint
    plus an optional docker container. Taken for vLLM profiles, for nodes
    with no current swap status, and for generic nodes where swap control
    itself is unconfigured (NODE_VLLM_DIR/NODE_SWAP_CTL_DIR unset)."""
    result = {"model": None, "endpoint_ok": False, "container_status": None}
    if nodeconfig.NODE_SERVING_CONTAINER:
        result["container_status"] = _container_status(
            nodeconfig.NODE_SERVING_CONTAINER)
    if not nodeconfig.NODE_SERVING_PROBE_URL:
        return result
    try:
        payload = _fetch_models_payload(nodeconfig.NODE_SERVING_PROBE_URL)
        models = payload.get("data") or []
        if models:
            result["model"] = models[0].get("id")
        result["endpoint_ok"] = True
    except (ProbeError, AttributeError, TypeError, KeyError, IndexError):
        pass
    return result


def probe_url_warning() -> str | None:
    """PROBE_URL_WARNING when the node config is blind, else None.

    Keyed on the configured profile LIST, not the active profile — a node
    currently serving ds4 with a vllm profile configured is one blind future
    swap away from the incident, so the warning must not wait for the swap.
    SwapCtlDisabled (generic node, no profiles) means there is nothing to
    know, not a misconfiguration."""
    if nodeconfig.NODE_SERVING_PROBE_URL:
        return None
    try:
        profiles = swapctl.list_profiles()
    except swapctl.SwapCtlDisabled:
        return None
    except OSError:
        # A profiles dir that raises on inspection (root-owned, not
        # traversable, ...) means the warning is UNDETERMINABLE — and a
        # warning-only feature must never take down what it decorates:
        # this runs on every serving poll and at agent boot, and the same
        # config booted and probed fine before the warning existed.
        return None
    if any(p.get("engine") == "vllm" for p in profiles):
        return PROBE_URL_WARNING
    return None


def log_startup_warning() -> None:
    """One startup log line, same condition and text as the probe field
    (called once at app import, node-agent app.py)."""
    warning = probe_url_warning()
    if warning is not None:
        _logger.warning(warning)


def _with_warning(result: dict) -> dict:
    """Attach the node-config warning to a probe result. All probe paths
    funnel through this: the field describes the node's configuration, not
    the currently-active profile."""
    warning = probe_url_warning()
    if warning is not None:
        result["warning"] = warning
    return result


def probe() -> dict:
    try:
        meta = swapctl.current_profile_meta()
    except swapctl.SwapCtlDisabled:
        # Generic node: no NODE_VLLM_DIR/NODE_SWAP_CTL_DIR configured, so
        # there is no per-profile metadata to consult at all -- fall back to
        # the env-configured path exactly as before swap control existed.
        meta = None

    if meta and meta.get("engine") != "vllm":
        container = meta.get("container") or nodeconfig.NODE_SERVING_CONTAINER
        result = {"model": meta["name"], "endpoint_ok": False,
                  "container_status": None}
        if container:
            result["container_status"] = _container_status(container)
        url = meta.get("health_url")
        if url:
            result["endpoint_ok"] = _probe_health_2xx(url)
        return _with_warning(result)

    return _with_warning(_probe_env_configured())
