"""tests/test_telemetry.py — LocalTelemetry: TTL-cached pass-through of
dashboard-api's own GET /api/gpu/detailed for the deck's local node entry.

Client is injected (FakeClient below), never a real httpx transport: this
class talks to a sibling container by DNS name, and no test here should
depend on Docker networking to pass.
"""

import types

import httpx

from app.telemetry import LocalTelemetry

_GPU_ROW = {
    "index": 0, "uuid": "u", "name": "AMD GPU", "memory_used_mb": 100,
    "memory_total_mb": 32624, "memory_percent": 0.3,
    "utilization_percent": 3, "temperature_c": 29, "power_w": 15.0,
    "assigned_services": ["llama_server"],   # must be DROPPED (install-time fiction)
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"gpus": [_GPU_ROW]}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


class FakeClient:
    """Records every call and hands back one canned response (or, if
    ``exc`` is given, raises it instead of ever answering)."""

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def get(self, path, headers=None):
        self.calls.append((path, headers))
        if self._exc is not None:
            raise self._exc
        return self._response


def _settings(url="http://dashboard-api:3002", key=""):
    return types.SimpleNamespace(dashboard_api_url=url, dashboard_api_key=key)


def test_gpus_passes_through_allowed_keys_only():
    client = FakeClient(FakeResponse())
    telemetry = LocalTelemetry(_settings(), client=client)

    rows = telemetry.gpus()

    assert rows == [{
        "index": 0, "uuid": "u", "name": "AMD GPU", "memory_used_mb": 100,
        "memory_total_mb": 32624, "memory_percent": 0.3,
        "utilization_percent": 3, "temperature_c": 29, "power_w": 15.0,
    }]
    assert "assigned_services" not in rows[0]


def test_gpus_returns_none_on_transport_error():
    client = FakeClient(exc=httpx.ConnectError("down"))
    telemetry = LocalTelemetry(_settings(), client=client)

    assert telemetry.gpus() is None


def test_gpus_returns_none_on_bad_payload():
    # Non-list "gpus" (TypeError, raised explicitly) and a payload missing
    # the "gpus" key entirely (KeyError) must both degrade to None.
    bad_shape = LocalTelemetry(_settings(),
                               client=FakeClient(FakeResponse(payload={"gpus": "nope"})))
    missing_key = LocalTelemetry(_settings(),
                                 client=FakeClient(FakeResponse(payload={})))

    assert bad_shape.gpus() is None
    assert missing_key.gpus() is None


def test_gpus_ttl_caches():
    client = FakeClient(FakeResponse())
    now = {"t": 0.0}
    telemetry = LocalTelemetry(_settings(), client=client, clock=lambda: now["t"])

    telemetry.gpus()
    telemetry.gpus()
    assert len(client.calls) == 1  # second call inside the TTL window: no fetch

    now["t"] += 10.0
    telemetry.gpus()
    assert len(client.calls) == 2  # TTL elapsed: fetches again


def test_auth_header_only_when_key_set():
    keyed = FakeClient(FakeResponse())
    LocalTelemetry(_settings(key="secret"), client=keyed).gpus()
    assert keyed.calls[0][1] == {"Authorization": "Bearer secret"}

    unkeyed = FakeClient(FakeResponse())
    LocalTelemetry(_settings(key=""), client=unkeyed).gpus()
    assert unkeyed.calls[0][1] == {}


def test_unconfigured_url_returns_none_without_fetch():
    client = FakeClient(FakeResponse())
    telemetry = LocalTelemetry(_settings(url=""), client=client)

    assert telemetry.gpus() is None
    assert client.calls == []
