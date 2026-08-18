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
    "memory_usage_available": True, "utilization_available": True,
    "temperature_available": True,
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


def _settings(url="http://dashboard-api:3002", key="", key_file="/ods-data/nope.txt"):
    return types.SimpleNamespace(dashboard_api_url=url, dashboard_api_key=key,
                                 dashboard_api_key_file=key_file)


def test_gpus_passes_through_allowed_keys_only():
    client = FakeClient(FakeResponse())
    telemetry = LocalTelemetry(_settings(), client=client)

    rows = telemetry.gpus()

    assert rows == [{
        "index": 0, "uuid": "u", "name": "AMD GPU", "memory_used_mb": 100,
        "memory_total_mb": 32624, "memory_percent": 0.3,
        "utilization_percent": 3, "temperature_c": 29, "power_w": 15.0,
        "memory_usage_available": True, "utilization_available": True,
        "temperature_available": True,
    }]
    assert "assigned_services" not in rows[0]


def test_availability_flags_are_carried_so_a_failed_sensor_is_not_a_reading():
    # dashboard-api sends value 0 + flag False on a sensor it could not read
    # (dashboard-api/gpu.py:172 `temperature_available=temp > 0`). Drop the
    # flag and the board renders a real 0 degC / 0% — the F1 defect. The UI's
    # own fold lives in ui/src/model/nodes.ts's statsOf.
    dark = {**_GPU_ROW, "temperature_c": 0, "temperature_available": False,
            "utilization_percent": 0, "utilization_available": False,
            "memory_used_mb": 0, "memory_usage_available": False}
    client = FakeClient(FakeResponse(payload={"gpus": [dark]}))

    row = LocalTelemetry(_settings(), client=client).gpus()[0]

    assert row["temperature_available"] is False
    assert row["utilization_available"] is False
    assert row["memory_usage_available"] is False


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


def test_a_failed_fetch_is_cached_for_the_ttl_too():
    # The module docstring's own claim ("a FAILED fetch is cached too, not
    # just a successful one — a dead dashboard-api must not be hammered on
    # every poll"), pinned. Every browser tab polling /api/state costs one
    # connection attempt per TTL window, not one per poll.
    client = FakeClient(exc=httpx.ConnectError("down"))
    now = {"t": 0.0}
    telemetry = LocalTelemetry(_settings(), client=client, clock=lambda: now["t"])

    assert telemetry.gpus() is None
    assert telemetry.gpus() is None
    assert len(client.calls) == 1  # inside the window: no second attempt

    now["t"] += 10.0
    assert telemetry.gpus() is None
    assert len(client.calls) == 2  # TTL elapsed: retried


def test_a_failure_is_logged_once_per_state_change(caplog):
    # Repo CLAUDE.md: a tolerated failure must be logged. Once per state
    # CHANGE, not once per 5 s poll — a dashboard-api that has been down for
    # a day must not have written 17k identical lines.
    client = FakeClient(exc=httpx.ConnectError("down"))
    now = {"t": 0.0}
    telemetry = LocalTelemetry(_settings(), client=client, clock=lambda: now["t"])

    with caplog.at_level("WARNING", logger="app.telemetry"):
        for _ in range(3):
            telemetry.gpus()
            now["t"] += 10.0
    assert len(caplog.records) == 1
    assert "dashboard-api" in caplog.records[0].getMessage()


def test_recovery_relogs_the_next_failure(caplog):
    # State CHANGE, both ways: a fetch that succeeds again re-arms the log,
    # so a flapping dashboard-api is visible rather than silent after the
    # first line.
    responses = [None, FakeResponse(), None]

    class Flapping:
        def __init__(self):
            self.calls = []

        def get(self, path, headers=None):
            self.calls.append((path, headers))
            resp = responses[len(self.calls) - 1]
            if resp is None:
                raise httpx.ConnectError("down")
            return resp

    now = {"t": 0.0}
    telemetry = LocalTelemetry(_settings(), client=Flapping(), clock=lambda: now["t"])

    with caplog.at_level("WARNING", logger="app.telemetry"):
        for _ in range(3):
            telemetry.gpus()
            now["t"] += 10.0
    assert len(caplog.records) == 2


def test_auth_header_only_when_key_set():
    keyed = FakeClient(FakeResponse())
    LocalTelemetry(_settings(key="secret"), client=keyed).gpus()
    assert keyed.calls[0][1] == {"Authorization": "Bearer secret"}

    unkeyed = FakeClient(FakeResponse())
    LocalTelemetry(_settings(key=""), client=unkeyed).gpus()
    assert unkeyed.calls[0][1] == {}


# --- stock-install auth (F3) ------------------------------------------------
# dashboard-api's /api/gpu/detailed REQUIRES a bearer (its security.py has no
# unauthenticated path). With DASHBOARD_API_KEY unset — the stock install —
# dashboard-api MINTS a random key into /data/dashboard-api-key.txt, and
# anything that does not read that file 401s forever. The dashboard's own
# nginx entrypoint already does exactly this (extensions/services/dashboard/
# entrypoint.sh:5-20); this is the same fallback, over the deck's ro
# /ods-data mount.


def test_key_file_is_read_when_the_env_key_is_empty(tmp_path):
    key_file = tmp_path / "dashboard-api-key.txt"
    key_file.write_text("minted-key\n")   # dashboard-api writes it with a newline
    client = FakeClient(FakeResponse())

    LocalTelemetry(_settings(key="", key_file=str(key_file)), client=client).gpus()

    assert client.calls[0][1] == {"Authorization": "Bearer minted-key"}


def test_env_key_wins_over_the_key_file(tmp_path):
    key_file = tmp_path / "dashboard-api-key.txt"
    key_file.write_text("minted-key")
    client = FakeClient(FakeResponse())

    LocalTelemetry(_settings(key="from-env", key_file=str(key_file)), client=client).gpus()

    assert client.calls[0][1] == {"Authorization": "Bearer from-env"}


def test_missing_key_file_sends_no_header(tmp_path):
    # Unmounted /ods-data, or a dashboard-api that has not started yet: no
    # header, exactly as before this fallback existed.
    client = FakeClient(FakeResponse())

    LocalTelemetry(_settings(key="", key_file=str(tmp_path / "absent.txt")),
                   client=client).gpus()

    assert client.calls[0][1] == {}


def test_unreadable_key_file_sends_no_header(tmp_path):
    # A directory where the file should be: OSError, same degradation.
    (tmp_path / "dashboard-api-key.txt").mkdir()
    client = FakeClient(FakeResponse())

    LocalTelemetry(_settings(key="", key_file=str(tmp_path / "dashboard-api-key.txt")),
                   client=client).gpus()

    assert client.calls[0][1] == {}


def test_key_file_appearing_later_is_picked_up_without_a_restart(tmp_path):
    # The real boot race: model-deck and dashboard-api start together, and
    # the key file does not exist yet when the deck builds its client. Read
    # per fetch (at most once per TTL) rather than once at construction, so
    # the deck heals itself instead of 401ing until someone restarts it.
    key_file = tmp_path / "dashboard-api-key.txt"
    client = FakeClient(FakeResponse())
    now = {"t": 0.0}
    telemetry = LocalTelemetry(_settings(key="", key_file=str(key_file)),
                               client=client, clock=lambda: now["t"])

    telemetry.gpus()
    assert client.calls[0][1] == {}

    key_file.write_text("minted-later")
    now["t"] += 10.0
    telemetry.gpus()
    assert client.calls[1][1] == {"Authorization": "Bearer minted-later"}


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
