"""The deck bracket: announce a deliberate teardown, then adopt.

These tests never touch a real deck. They assert the CALL SEQUENCE, because
the sequence is the contract: hold before the teardown, adopt after it ONLY
if the body committed (proved the new state serving), and release the hold
whenever no adopt happened.
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
    with agent._deck_bracket({}, "local/hipfire") as bracket:
        bracket.commit()

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

    with agent._deck_bracket({}, "local/hipfire") as bracket:
        bracket.commit()

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
    """The bracket yields a handle. `renew()` re-arms the same hold with the
    same TTL — used before a second teardown (e.g. a rollback recreate) so it
    is not racing the first hold's expiry."""
    with agent._deck_bracket({}, "local/hipfire", ttl_s=250) as bracket:
        bracket.renew()
        bracket.commit()

    assert [(m, p, pl) for m, p, pl in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire", {"ttl_s": 250}),
        ("POST", "/api/lifecycle/expect-absence/local/hipfire", {"ttl_s": 250}),
        ("POST", "/api/lifecycle/adopt/local/hipfire", None),
    ]


def test_happy_path_when_renew_is_never_called_is_still_just_hold_then_adopt(calls):
    """Renew is opt-in per call site. A caller that never invokes it gets
    exactly the original hold-then-adopt sequence, unchanged."""
    with agent._deck_bracket({}, "local/hipfire") as bracket:
        bracket.commit()

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("POST", "/api/lifecycle/adopt/local/hipfire"),
    ]


def test_no_commit_means_no_adopt_and_the_hold_is_released(calls):
    """THE C1 GUARD. A body that finished without proving the new state is
    serving — the rollback path, or any early return before the health gate
    closes — must not adopt.

    Adopt would record what the deck can see right now, and right after a
    rollback recreate that is a container which is up but still loading:
    reachable, not loaded, i.e. state="unloaded" actor="operator". That is a
    deliberate park, and a parked hipfire is one plan_reconcile never
    restores again — the failure the reconciler exists to prevent, written
    into intent.json by the tool meant to prevent it.
    """
    with agent._deck_bracket({}, "local/hipfire") as bracket:
        bracket.renew()          # the rollback path re-arms, then does NOT commit

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("DELETE", "/api/lifecycle/expect-absence/local/hipfire"),
    ]


def test_no_commit_still_fails_open(calls):
    """The other half of the no-commit path: the hold is RELEASED, not left
    to time out. A best-effort bracket that concluded nothing must never
    leave the deck's reconciler standing down for the full TTL."""
    with agent._deck_bracket({}, "local/hipfire"):
        pass

    assert calls[-1][0] == "DELETE"
    assert "adopt" not in " ".join(p for _, p, _ in calls)


def test_commit_is_what_earns_the_adopt(calls):
    """Same body, same bracket, one extra call — and only then an adopt.
    Paired with the two tests above so the difference is visibly the commit
    and nothing else."""
    with agent._deck_bracket({}, "local/hipfire") as bracket:
        bracket.renew()
        bracket.commit()

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("POST", "/api/lifecycle/adopt/local/hipfire"),
    ]


def test_an_exception_after_commit_still_adopts(calls):
    """Committed means the engine was PROVED serving. If the body then blows
    up (json_response failing after the health gate and the LiteLLM restart
    both passed), the new model really is up and the deck should record it —
    the exception is about reporting, not about what is running."""
    with pytest.raises(RuntimeError):
        with agent._deck_bracket({}, "local/hipfire") as bracket:
            bracket.commit()
            raise RuntimeError("reporting the success failed")

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("DELETE", "/api/lifecycle/expect-absence/local/hipfire"),
    ]


def test_deck_call_reports_an_http_refusal_without_a_traceback(monkeypatch, caplog):
    """urlopen raises HTTPError for EVERY 4xx/5xx, so a non-2xx never reaches
    a status check after urlopen returns — the friendly branch that used to
    live there was dead code and every refusal landed in the broad handler
    with a full traceback. 409 is routine now that adopt refuses unreachable
    and mid-transition resources by design, so it is logged as a plain
    warning naming the code."""
    import urllib.error

    def refuse(*a, **kw):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:3015/api/lifecycle/adopt/local/hipfire",
            409, "Conflict", {}, None)

    monkeypatch.setattr(agent.urllib_request, "urlopen", refuse)

    with caplog.at_level("WARNING"):
        assert agent._deck_call({}, "POST", "/api/lifecycle/adopt/local/hipfire") is False

    record = caplog.records[-1]
    assert "409" in record.getMessage()
    assert record.exc_info is None
