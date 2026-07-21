"""
Lemonade engine client — model load/unload, status, and idle-activity metrics.

Talks to Lemonade's REST API (default http://host.docker.internal:13305)
over a 5 s httpx.Client, authenticated with `Authorization: Bearer <key>`
(verified live). A `transport=` kwarg lets tests inject httpx.MockTransport
instead of hitting the network.

Real response shapes this is coded against:
  GET  /api/v1/health -> {"model_loaded": "extra.Qwen3.5-27B-Q4_K_M.gguf"}
                       or {"model_loaded": null}  (other keys may exist; ignored)
  POST /api/v1/load    body {"model_name": "<name>"}
  POST /api/v1/unload  body {"model_name": "<name>"}
  GET  /metrics         Prometheus text exposition format

status()/load()/unload() raise EngineError on a non-2xx response (with the
response text) or on an httpx.TransportError (e.g. connection refused).

activity() is the one method that does not raise: any transport error,
non-2xx response, or a metrics page with neither target metric present
returns None so callers can disable idle-TTL tracking and log once,
rather than crash-looping on a Lemonade that's merely unreachable.
"""

import httpx

from app.engines import EngineError

_TIMEOUT = 5.0

# Prometheus metric name prefixes summed to approximate total engine
# activity since last scrape (llama.cpp counters, monotonically increasing).
_ACTIVITY_METRIC_PREFIXES = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:tokens_predicted_total",
)


class LemonadeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        metrics_url: str | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=_TIMEOUT,
            transport=transport,
        )
        # Lemonade's wrapper serves its web UI at /metrics; the wrapped
        # llama-server exposes the real Prometheus counters on its own port
        # (e.g. http://llama-server:8001/metrics). Default keeps the old
        # base-relative path for setups without a separate metrics port.
        self._metrics_url = metrics_url or f"{base_url.rstrip('/')}/metrics"

    def status(self) -> dict:
        resp = self._request("GET", "/api/v1/health")
        return {"loaded": resp.json().get("model_loaded")}

    def load(self, model_name: str) -> None:
        self._request("POST", "/api/v1/load", json={"model_name": model_name})

    def unload(self, model_name: str) -> None:
        self._request("POST", "/api/v1/unload", json={"model_name": model_name})

    def activity(self) -> int | None:
        try:
            resp = self._client.get(self._metrics_url)
        except httpx.TransportError:
            return None
        if not resp.is_success:
            return None
        return _sum_activity_metrics(resp.text)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp


def _sum_activity_metrics(metrics_text: str) -> int | None:
    """Sum values of Prometheus lines whose metric name starts with one of
    _ACTIVITY_METRIC_PREFIXES. Returns None if neither metric appears, or on
    any metric value parse error."""
    total = 0.0
    found = False
    try:
        for line in metrics_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, value = parts
            if name.startswith(_ACTIVITY_METRIC_PREFIXES):
                total += float(value)
                found = True
    except ValueError:
        return None
    return int(round(total)) if found else None
