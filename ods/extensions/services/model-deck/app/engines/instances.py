"""Deck -> node-agent INSTANCE verbs. Thin and dumb by design (same posture
as app.engines.sglang_omni): one POST per verb, 202 means queued for the
host-side instances-helper, no waiting, no retries (the router decides what a
refusal means). No "local" anywhere -- the node is whatever registry entry
declared control:"instances"."""
from app.engines import BusyError, EngineError, engine_request
from app.engines.node_agent import NodeAgentHTTP

_ACCEPTED = 202


class InstancesClient(NodeAgentHTTP):
    def request(self, verb: str, document: dict) -> None:
        resource = document["resource"]
        try:
            resp = engine_request(lambda: self._node.post(
                f"/v1/node/instance/{resource}", json={"verb": verb, "document": document}),
                also_ok=(409,))
        except EngineError as exc:
            raise type(exc)(f"instance {resource!r} {verb}: {exc}") from exc
        if resp.status_code == 409:
            raise BusyError(f"instance {resource!r} {verb}: {resp.text}")
        if resp.status_code != _ACCEPTED:
            raise EngineError(f"instance {resource!r} {verb} not accepted (status {resp.status_code}): {resp.text}")

    def status(self, resource: str) -> dict | None:
        return self._node_get(f"/v1/node/instance/{resource}/status").get("result")
