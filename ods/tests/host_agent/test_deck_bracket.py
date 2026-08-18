"""The deck bracket: announce a deliberate teardown, then adopt.

These tests never touch a real deck. They assert the CALL SEQUENCE, because
the sequence is the contract: hold before the teardown, adopt after it ONLY
if the body committed (proved the new state serving), and release the hold
whenever no adopt happened.
"""

import ast
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


def test_an_exception_after_commit_still_fails_open(calls):
    """commit() only arms the adopt — it does not run it. `yield handle` sits
    between `commit()` and the `if handle.committed:` adopt call
    (`ods/bin/ods-host-agent.py:12608`); an exception raised in the body
    unwinds straight past that `if` into `finally`, so adopt never fires and
    the hold is released instead. Committed-then-raised fails open exactly
    like never-committed does.

    That is safe: a missed adopt leaves the deck merely unrecorded about a
    resource that in fact came up — the pre-branch state, never a false
    park. A false park (recording state="unloaded" for something running)
    is the failure this branch exists to prevent; an unrecorded-but-running
    resource is not that.

    This path is not reachable in `_do_hipfire_activate` today: its `except`
    handler returns early once `committed` is true
    (`ods-host-agent.py:8964-8966`) instead of re-raising, so the caller
    never lets an exception escape the body after commit(). That early
    return is load-bearing for this test's safety argument — if it were
    ever removed, a proved-serving activation could still lose its adopt
    here, and the resource would sit unmanaged (not falsely parked, but
    invisible to the deck) until some other actuation happened to adopt it.
    """
    with pytest.raises(RuntimeError):
        with agent._deck_bracket({}, "local/hipfire") as bracket:
            bracket.commit()
            raise RuntimeError("reporting the success failed")

    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/api/lifecycle/expect-absence/local/hipfire"),
        ("DELETE", "/api/lifecycle/expect-absence/local/hipfire"),
    ]


def _activate_def():
    """The ast.FunctionDef for _do_model_activate, parsed fresh from source.

    These wiring pins are STRUCTURAL on purpose. `_do_model_activate` is a
    ~1100-line handler whose realistic states cannot be constructed in a
    unit test (the hipfire wave's Task 4 ledgered exactly this gap: the
    bracket wiring there is "verified by reading only"). An AST pin is the
    honest middle ground — it cannot prove behaviour, but it makes the
    load-bearing placement facts (which key, what the bracket encloses,
    where commit and renew sit) fail loudly if a refactor moves them.
    """
    tree = ast.parse(_AGENT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_do_model_activate":
            return node
    raise AssertionError("_do_model_activate not found in ods-host-agent.py")


def _bracket_with(func):
    """The `with _deck_bracket(...)` node inside a function, or None."""
    for node in ast.walk(func):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_deck_bracket"):
                return node, call
    return None, None


def test_model_activate_brackets_its_transaction_under_the_lemonade_key():
    """THE LEMONADE GAP. `_do_model_activate` tears down ods-llama-server on
    its container strategies (compose stop+up, or inspect-and-recreate) and
    again on rollback — the deck's declared `lemonade` resource, which is
    arbiter-eligible, so the race here is automatic, not operator-vs-deck.
    The whole try/except must sit inside a bracket keyed "local/lemonade"
    (node-qualified: live intent.json keys are "local/hipfire",
    "sparky/slot0"), so the except handler's rollback recreate is covered
    too, exactly as in _do_hipfire_activate."""
    func = _activate_def()
    with_node, call = _bracket_with(func)
    assert with_node is not None, "_do_model_activate has no _deck_bracket with-block"
    assert isinstance(call.args[1], ast.Constant) and call.args[1].value == "local/lemonade"

    # The bracket must enclose the MAIN transaction try/except — the one
    # whose handler rolls back via rollback_and_prove() — not some inner try.
    def calls_rollback(node):
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "rollback_and_prove"
            for n in ast.walk(node)
        )

    assert any(
        isinstance(stmt, ast.Try)
        and any(calls_rollback(handler) for handler in stmt.handlers)
        for stmt in ast.walk(with_node)
    ), "the bracket does not enclose the try/except that rolls back"


def test_model_activate_rollback_renews_the_hold_before_restoring():
    """Both rollback entries (health-fail else-branch and the except handler)
    funnel through rollback_and_prove(), whose restore recreates the
    container a second time — after a health wait that can consume most of
    the original TTL. Renew must be its first act, before any restore."""
    func = _activate_def()
    rollback = next(
        (n for n in ast.walk(func)
         if isinstance(n, ast.FunctionDef) and n.name == "rollback_and_prove"),
        None,
    )
    assert rollback is not None

    def line_of_call(node, attr=None, name=None):
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            if attr and isinstance(n.func, ast.Attribute) and n.func.attr == attr:
                return n.lineno
            if name and isinstance(n.func, ast.Name) and n.func.id == name:
                return n.lineno
        return None

    renew_line = line_of_call(rollback, attr="renew")
    restore_line = line_of_call(rollback, name="restore_backups")
    assert renew_line is not None, "rollback_and_prove never renews the hold"
    assert restore_line is not None
    assert renew_line < restore_line, "renew must precede the restore"


def test_model_activate_commits_exactly_at_the_committed_flag():
    """commit() is what earns the adopt, and it must mean the same thing the
    handler's own `committed = True` means: final runtime proof passed and
    the receipt is written. Pin them adjacent so they can never drift."""
    func = _activate_def()
    commits = [
        n for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "commit"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "deck_bracket"
    ]
    assert len(commits) == 1, "expected exactly one deck_bracket.commit()"

    flag_lines = [
        n.lineno for n in ast.walk(func)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "committed" for t in n.targets)
        and isinstance(n.value, ast.Constant) and n.value.value is True
    ]
    assert len(flag_lines) == 1
    # Adjacent: the commit call sits on the line right after `committed = True`
    # (allowing the flag's trailing comment to share its line).
    assert commits[0].lineno == flag_lines[0] + 1


def _renew_lines(node):
    return [
        n.lineno for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "renew"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "deck_bracket"
    ]


def test_model_activate_hold_ttl_is_the_maximum():
    """900s (MAX_HOLD_TTL_S) — not the 600s default. Unlike hipfire, this
    handler's single longest stage (the readiness wait, ~600s+ worst case)
    can exceed the default TTL on its own, so even with stage-boundary
    renewals each window must be sized to the largest single stage."""
    _, call = _bracket_with(_activate_def())
    ttl = next((kw.value for kw in call.keywords if kw.arg == "ttl_s"), None)
    if ttl is None and len(call.args) > 2:
        ttl = call.args[2]
    assert isinstance(ttl, ast.Constant) and ttl.value == 900


def test_model_activate_renews_at_every_forward_stage_boundary():
    """The TTL bounds a STUCK actuator, not the whole transaction. The
    forward path is restart caps (~420s) + readiness wait (~600s+) +
    dependent-container health waits (~600s) before commit(); armed once at
    entry, the hold would lapse mid-activation and reopen the automatic
    race for exactly the activations big enough to matter. Four renewals,
    each at a stage boundary the handler already has:

      1. before the runtime-restart section
      2. first statement of _sb_wait_ready (the switchboard path's wait —
         restart and wait both happen inside run_runtime_activation, so
         this callback is the only seam between them)
      3. first statement of the non-switchboard readiness-wait branch
      4. first statement of the `if healthy:` success branch (before the
         dependent restarts and final proof)
    """
    func = _activate_def()
    with_node, _ = _bracket_with(func)

    # 2. _sb_wait_ready opens with a renew.
    sb_wait = next(
        n for n in ast.walk(func)
        if isinstance(n, ast.FunctionDef) and n.name == "_sb_wait_ready"
    )
    assert sb_wait.body[0].lineno in _renew_lines(sb_wait), \
        "_sb_wait_ready's first statement is not deck_bracket.renew()"

    # Locate the runtime-restart dispatch by its `_in_container` assignment.
    in_container_line = next(
        n.lineno for n in ast.walk(func)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_in_container" for t in n.targets)
    )

    # 1. A renew inside the bracket, before the restart section (the
    # rollback-side renew lives in rollback_and_prove, defined earlier —
    # exclude it by requiring the line to be inside the with-block).
    forward = [l for l in _renew_lines(with_node)]
    assert any(with_node.lineno < l < in_container_line for l in forward), \
        "no renew between bracket entry and the runtime-restart section"

    # 3. The non-switchboard branch (the orelse holding the big readiness
    # wait) opens with a renew.
    def assigns_runtime_identity(stmts):
        return any(
            isinstance(a, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "runtime_identity" for t in a.targets)
            for stmt in stmts for a in ast.walk(stmt)
        )

    sb_if = next(
        n for n in ast.walk(func)
        if isinstance(n, ast.If) and n.orelse and assigns_runtime_identity(n.orelse)
    )
    assert sb_if.orelse[0].lineno in _renew_lines(sb_if), \
        "the readiness-wait branch does not open with a renew"

    # 4. The `if healthy:` success branch opens with a renew.
    healthy_if = next(
        n for n in ast.walk(func)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == "healthy"
        and any(
            isinstance(a, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "final_runtime_proof" for t in a.targets)
            for a in ast.walk(n)
        )
    )
    assert healthy_if.body[0].lineno in _renew_lines(healthy_if), \
        "the success branch does not open with a renew"


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
