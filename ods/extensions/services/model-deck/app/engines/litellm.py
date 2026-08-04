"""
litellm engine client — route table introspection.

Talks to the litellm proxy's REST API (default http://litellm:4000) over a
5 s httpx.Client, authenticated with `Authorization: Bearer <key>`
(verified live, same scheme as Lemonade). A `transport=` kwarg lets tests
inject httpx.MockTransport instead of hitting the network.

Real response shape this is coded against (GET /model/info):
  {"data": [
    {"model_name": "default",  "litellm_params": {"model": "openai/...", "api_base": "..."}},
    {"model_name": "hipfire",  "litellm_params": {"model": "openai/...", "api_base": "..."}},
    {"model_name": "lemonade", "litellm_params": {...}},
    {"model_name": "*",        "litellm_params": {...}},
  ]}

Both methods raise EngineError on a non-2xx response (with the response
text) or on an httpx.TransportError.
"""

import httpx

from app.engines import EngineError

_TIMEOUT = 5.0


class LiteLLMClient:
    def __init__(
        self,
        base_url: str,
        key: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=_TIMEOUT,
            transport=transport,
        )

    def route_table(self) -> dict[str, str]:
        entries = self._model_info()
        return {entry["model_name"]: entry["litellm_params"]["model"] for entry in entries}

    def default_targets_hipfire(self) -> bool:
        for entry in self._model_info():
            if entry["model_name"] == "default":
                return "hipfire" in entry["litellm_params"].get("api_base", "")
        return False

    def model_info(self) -> list[dict]:
        """Raw /model/info entries — for callers needing api_base etc."""
        return self._model_info()

    def health(self) -> dict:
        """Raw GET /health body: healthy_endpoints / unhealthy_endpoints.

        Note this is the *proxy's* per-route probe, which actively calls each
        backend. Interpretation lives in ``interpret_health`` — a non-2xx or
        transport failure here raises EngineError like every other call.
        """
        try:
            resp = self._client.get("/health")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()

    def _model_info(self) -> list[dict]:
        try:
            resp = self._client.get("/model/info")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()["data"]


# Substrings identifying "the node answered, it just isn't serving this
# model" — the normal resting state of every single-slot engine. Anything
# else is treated as unreachable: an unrecognised error must fail loud
# rather than be silently downgraded to a benign reading.
_NOT_LOADED_MARKERS = ("does not exist", "notfounderror", "model_not_found")


def interpret_health(body: dict) -> dict[str, dict]:
    """Map a /health body to ``{route: {reachable, loaded, detail}}``.

    The load-bearing rule (design doc, 'LiteLLM interpretation rule'): a
    connection error means the node is DOWN; a model-not-found error on a
    reachable node means NOT LOADED and is never an alarm. On 2026-08-03,
    five of six 'unhealthy' routes were the latter and entirely correct.
    """
    result: dict[str, dict] = {}

    for entry in body.get("healthy_endpoints") or []:
        result[entry["model"]] = {"reachable": True, "loaded": True, "detail": "healthy"}

    for entry in body.get("unhealthy_endpoints") or []:
        error = str(entry.get("error", ""))
        not_loaded = any(marker in error.lower() for marker in _NOT_LOADED_MARKERS)
        result[entry["model"]] = {
            "reachable": not_loaded,
            "loaded": False,
            "detail": error or "unknown error",
        }

    return result
