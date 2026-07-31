"""Probe what this node is serving (OpenAI-compatible endpoint + container)."""
import subprocess

import httpx

import nodeconfig
import swapctl


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
        return result

    return _probe_env_configured()
