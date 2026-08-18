"""The deck bracket: announce a deliberate teardown, then adopt.

These tests never touch a real deck. They assert the CALL SEQUENCE, because
the sequence is the contract: hold before the teardown, adopt after it, and
release if adopt never happened.
"""

import http.client
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


def test_deck_call_survives_an_unserializable_payload():
    """json.dumps raises TypeError on a payload it cannot serialize (e.g. a
    bare object()). _deck_call must never raise — a raise here would replace
    whatever exception is in flight and, called from the bracket's `except`
    handler, would skip the rollback entirely. No stubbing: the real
    json.dumps does the raising."""
    assert agent._deck_call({}, "POST", "/api/lifecycle/expect-absence/local/hipfire",
                             {"bad": object()}) is False


def test_deck_call_survives_http_client_exceptions(monkeypatch):
    """http.client.HTTPException (e.g. BadStatusLine, InvalidURL) is not an
    OSError or a ValueError. A deck answering with garbage — or a malformed
    MODEL_DECK_PORT pointing at a non-HTTP listener — must not escape
    _deck_call and kill the caller."""
    def explode(*a, **kw):
        raise http.client.BadStatusLine("")

    monkeypatch.setattr(agent.urllib_request, "urlopen", explode)

    assert agent._deck_call({}, "POST", "/api/lifecycle/expect-absence/local/hipfire",
                             {"ttl_s": 10}) is False


def test_a_down_deck_never_breaks_the_caller_on_http_client_exceptions(monkeypatch):
    """Same as test_a_down_deck_never_breaks_the_caller, but for the
    HTTPException family specifically — this is the exact hole the fix
    round closed."""
    def explode(*a, **kw):
        raise http.client.BadStatusLine("")

    monkeypatch.setattr(agent.urllib_request, "urlopen", explode)

    body_ran = False
    with agent._deck_bracket({}, "local/hipfire"):
        body_ran = True

    assert body_ran is True


def test_renew_reissues_the_hold(calls):
    """The bracket yields a callable. Calling it re-arms the same hold with
    the same TTL — used before a second teardown (e.g. a rollback recreate)
    so it is not racing the first hold's expiry."""
    with agent._deck_bracket({}, "local/hipfire", ttl_s=250) as renew:
        assert callable(renew)
        renew()

    assert [(m, p, pl) for m, p, pl in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire", {"ttl_s": 250}),
        ("POST", "/api/lifecycle/expect-absence/local/hipfire", {"ttl_s": 250}),
        ("POST", "/api/lifecycle/adopt/local/hipfire", None),
    ]


def test_happy_path_when_renew_is_never_called_is_still_just_hold_then_adopt(calls):
    """Renew is opt-in per call site. A caller that never invokes it gets
    exactly the original hold-then-adopt sequence, unchanged."""
    with agent._deck_bracket({}, "local/hipfire") as renew:
        assert callable(renew)

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("POST", "/api/lifecycle/adopt/local/hipfire"),
    ]
