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

from app.engines import engine_request

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

    def _model_info(self) -> list[dict]:
        resp = engine_request(lambda: self._client.get("/model/info"))
        return resp.json()["data"]
