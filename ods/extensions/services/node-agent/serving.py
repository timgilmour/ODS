"""Probe what this node is serving (OpenAI-compatible endpoint + container)."""
import subprocess

import httpx

import nodeconfig


class ProbeError(RuntimeError):
    pass


def _fetch_models_payload(url: str) -> dict:
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ProbeError(str(exc)) from exc


def _container_status(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True, text=True, timeout=2.0,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def probe() -> dict:
    result = {"model": None, "endpoint_ok": False, "container_status": None}
    if nodeconfig.NODE_SERVING_CONTAINER:
        result["container_status"] = _container_status(
            nodeconfig.NODE_SERVING_CONTAINER)
    if not nodeconfig.NODE_SERVING_PROBE_URL:
        return result
    try:
        payload = _fetch_models_payload(nodeconfig.NODE_SERVING_PROBE_URL)
        models = payload.get("data") or []
        result["endpoint_ok"] = True
        if models:
            result["model"] = models[0].get("id")
    except ProbeError:
        pass
    return result
