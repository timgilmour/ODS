"""HTTP API tests for the DEPRECATED /api/spark/* alias (N1 T10).

app/routers/spark.py is now a thin resolver: exactly one control:"swap"
node -> forward to app.routers.serving's handlers; none -> 503 with the
byte-exact legacy "spark engine is not configured" message (existing
feature-detecting callers keep working); several -> 409 naming the
candidates. All per-route behavior (passthrough, guard/busy/engine-error
mapping, intent recording, observation invalidation, reload's guards) now
lives on app/routers/serving.py and is tested in tests/test_serving_api.py
against the same underlying code (serving.py's handlers are what spark.py
forwards to). This file only tests the ALIAS's own resolution contract.

FakeSpark is kept here (unrelated to the alias rewrite): tests/test_api.py
and tests/test_serving_api.py import it as the fake client wired via
wire_swap_node for both the node registry/adopt suite and the per-node
serving suite.
"""

from fastapi.testclient import TestClient

from tests.test_api import HERETIC_COMPOSE, make_app, wire_swap_node


class FakeSpark:
    def __init__(self):
        self.calls = []  # mutating only: ("swap", profile, force)
        self.status_calls = 0
        self.fail = None
        self.settings_sent = None  # (profile, document), last put_settings call
        self.settings_fail = None
        # Reload re-fetches the profile's compose before shipping (final
        # branch review: a stale service name in the identity map would
        # introduce an imageless service AFTER teardown killed everything),
        # so every reload test needs real compose text behind get_compose.
        self.compose = {}          # {profile: text}; default = the fixture
        self.compose_fail = None
        self._status = {
            "profiles": [
                {"name": "laguna", "engine": "vllm", "health_url": None,
                 "container": None},
                {"name": "mm27b", "engine": "vllm", "health_url": None,
                 "container": None},
            ],
            "swap_status": None,
            "serving": {"model": "aeon", "endpoint_ok": True,
                        "container_status": None},
        }

    def status(self):
        self.status_calls += 1
        return self._status

    def swap(self, profile, force=False):
        self.calls.append(("swap", profile, force))
        if self.fail:
            raise self.fail
        return {"id": "u1", "profile": profile}

    def put_settings(self, profile, document):
        if self.settings_fail:
            raise self.settings_fail
        self.settings_sent = (profile, document)

    def get_compose(self, profile):
        if self.compose_fail:
            raise self.compose_fail
        return self.compose.get(profile, HERETIC_COMPOSE)


def test_alias_503_when_no_swap_nodes(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    r = TestClient(app).get("/api/spark/status")
    assert r.status_code == 503
    assert r.json()["detail"] == "spark engine is not configured"


def test_alias_resolves_single_swap_node(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    fake = FakeSpark()
    wire_swap_node(deck, "boxa", fake, label="Box Alpha")
    client = TestClient(app)
    assert client.get("/api/spark/status").status_code == 200
    client.post("/api/spark/swap", json={"profile": "laguna"})
    assert ("swap", "laguna", False) in fake.calls
    assert "boxa/slot0" in deck["intent_store"].get()


def test_alias_409s_with_two_swap_nodes(tmp_path, monkeypatch):
    """Mutation check (design §10): never guess between candidates."""
    app, deck = make_app(tmp_path, monkeypatch)
    wire_swap_node(deck, "boxa", FakeSpark(), label="Box Alpha")
    wire_swap_node(deck, "boxb", FakeSpark(), label="Box Beta")
    r = TestClient(app).get("/api/spark/status")
    assert r.status_code == 409
    assert "boxa" in r.json()["detail"] and "boxb" in r.json()["detail"]
