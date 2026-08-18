"""tests/test_telemetry.py — LocalTelemetry: TTL-cached pass-through of
dashboard-api's own GET /api/gpu/detailed for the deck's local node entry.

Client is injected (FakeClient below), never a real httpx transport: this
class talks to a sibling container by DNS name, and no test here should
depend on Docker networking to pass.
"""

import types

import httpx

from app.gpu import MIN_VRAM_BYTES
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


# --- index vocabulary -------------------------------------------------------
# dashboard-api enumerates every DRM card it can see, iGPUs included (live
# autarch: index 2 is a 2048 MB 0x13c0 display GPU). The deck's own
# app.gpu.read_gpus excludes anything under MIN_VRAM_BYTES and RE-SEQUENCES
# what survives (app/gpu.py:33, :110-113 — an excluded card "does not consume
# a slot"), so world.gpus holds 0,1 for the same box. The UI joins the two
# lists on that number (ui/src/model/nodes.ts's statsOf), so this pass-through
# has to speak world.gpus' vocabulary or a card's temp/power/name silently
# describes a different GPU.

def _row(index, total_mb, **over):
    return {**_GPU_ROW, "index": index, "memory_total_mb": total_mb, **over}


def test_drops_sub_threshold_rows_and_resequences_survivors():
    client = FakeClient(FakeResponse(payload={"gpus": [
        _row(0, 32624, uuid="disc-a"),
        _row(1, 32624, uuid="disc-b"),
        _row(2, 2048, uuid="igpu"),      # the live autarch iGPU: excluded
    ]}))

    rows = LocalTelemetry(_settings(), client=client).gpus()

    assert [r["index"] for r in rows] == [0, 1]
    assert [r["uuid"] for r in rows] == ["disc-a", "disc-b"]


def test_resequences_when_the_igpu_enumerates_first():
    # The case the raw index would get WRONG: with the small card at raw 0,
    # a straight pass-through would put the first discrete's readings on the
    # deck's GPU 1 and leave GPU 0 showing the iGPU's.
    client = FakeClient(FakeResponse(payload={"gpus": [
        _row(0, 2048, uuid="igpu"),
        _row(1, 32624, uuid="disc-a", temperature_c=71),
        _row(2, 32624, uuid="disc-b", temperature_c=29),
    ]}))

    rows = LocalTelemetry(_settings(), client=client).gpus()

    assert [(r["index"], r["uuid"], r["temperature_c"]) for r in rows] == [
        (0, "disc-a", 71), (1, "disc-b", 29)]


def test_qualifying_bar_is_the_deck_s_own_constant():
    # Not a literal 4 GiB re-typed here: exactly-at-the-bar qualifies and one
    # byte under does not, both read off app.gpu.MIN_VRAM_BYTES.
    at_bar = MIN_VRAM_BYTES // (1024 * 1024)
    client = FakeClient(FakeResponse(payload={"gpus": [
        _row(0, at_bar - 1, uuid="under"),
        _row(1, at_bar, uuid="at"),
    ]}))

    rows = LocalTelemetry(_settings(), client=client).gpus()

    assert [(r["index"], r["uuid"]) for r in rows] == [(0, "at")]
