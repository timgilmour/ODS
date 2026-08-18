"""The deck bracket: announce a deliberate teardown, then adopt.

These tests never touch a real deck. They assert the CALL SEQUENCE, because
the sequence is the contract: hold before the teardown, adopt after it, and
release if adopt never happened.
"""

import importlib.util
from pathlib import Path

import pytest

_AGENT_PATH = Path(__file__).resolve().parents[2] / "bin" / "ods-host-agent.py"


def _load_agent():
    spec = importlib.util.spec_from_file_location(
        "ods_host_agent_under_test", _AGENT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load_agent()


@pytest.fixture
def calls(monkeypatch):
    """Record every deck call and control its success.

    A list SUBCLASS, not a bare list: plain lists reject attribute
    assignment, so `recorded.outcome = ...` would raise AttributeError.
    """
    class Calls(list):
        outcome = True

    recorded = Calls()

    def fake_call(env, method, path, payload=None):
        recorded.append((method, path, payload))
        return recorded.outcome

    monkeypatch.setattr(agent, "_deck_call", fake_call)
    return recorded


def test_happy_path_holds_then_adopts(calls):
    with agent._deck_bracket({}, "local/hipfire"):
        pass

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("POST", "/api/lifecycle/adopt/local/hipfire"),
    ]


def test_the_hold_carries_the_ttl(calls):
    with agent._deck_bracket({}, "local/hipfire", ttl_s=120):
        pass

    assert calls[0][2] == {"ttl_s": 120}


def test_the_hold_precedes_the_body(calls):
    """Ordering is the whole point: a hold placed after the recreate is late."""
    seen = []

    with agent._deck_bracket({}, "local/hipfire"):
        seen.append(len(calls))

    assert seen == [1]


def test_an_exception_releases_the_hold_and_propagates(calls):
    with pytest.raises(RuntimeError):
        with agent._deck_bracket({}, "local/hipfire"):
            raise RuntimeError("recreate blew up")

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("DELETE", "/api/lifecycle/expect-absence/local/hipfire"),
    ]


def test_a_failed_adopt_falls_back_to_release(calls):
    calls.outcome = False

    with agent._deck_bracket({}, "local/hipfire"):
        pass

    assert [m for m, _, _ in calls] == ["POST", "POST", "DELETE"]


def test_a_down_deck_never_breaks_the_caller(monkeypatch):
    """Best-effort: the operator's model change outranks the deck knowing."""
    def explode(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(agent.urllib_request, "urlopen", explode)

    body_ran = False
    with agent._deck_bracket({}, "local/hipfire"):
        body_ran = True

    assert body_ran is True


def test_deck_base_url_reads_the_configured_port():
    assert agent._deck_base_url({"MODEL_DECK_PORT": "3999"}) == "http://127.0.0.1:3999"
    assert agent._deck_base_url({}) == "http://127.0.0.1:3015"
    assert agent._deck_base_url({"MODEL_DECK_PORT": ""}) == "http://127.0.0.1:3015"
