"""Tests for app.configure — how settings reach an engine.

The write boundary is what each engine DECLARES it can do, not local vs
remote. That is what makes 'the Deck controls any connected inference
source' a rule rather than a pile of special cases — and it keeps an honest
slot for sources that genuinely cannot be written.
"""

import pytest

from app.configure import MECHS, apply_settings
from app.engines import EngineError


class FakeApiEngine:
    def __init__(self):
        self.configured = None

    def configure(self, settings):
        self.configured = settings


class FakeRestartEngine:
    def __init__(self):
        self.env = None
        self.restarted = False

    def set_env(self, env):
        self.env = env

    def restart(self):
        self.restarted = True


def test_mech_vocabulary():
    assert set(MECHS) == {"api", "env+restart", "node-settings", "none"}


def test_api_mech_refuses_because_no_client_implements_it():
    """EXPECTATION CHANGED [max-review c1/c12]. This test used to assert the
    api mech applied live against FakeApiEngine.configure(). That passed only
    because the FAKE implements configure() — no client in app/engines does,
    so the real dispatch died as AttributeError deep inside a route, which
    reads as a deck bug rather than "this was never built".

    The mech stays in MECHS: the descriptors declare it, and deleting the
    vocabulary would be a bigger lie than admitting it is unbuilt. So the
    honest behaviour is an explicit refusal in the contract's own error
    type, and this test now pins that. The fake keeps its configure() to
    prove the refusal happens BEFORE any dispatch — an implementation that
    called through would set `configured` and fail the last assertion.
    """
    engine = FakeApiEngine()

    with pytest.raises(EngineError, match="api"):
        apply_settings("api", engine_client=engine, resolved={"a": {"value": "1"}})

    assert engine.configured is None


def test_env_restart_mech_refuses_because_no_client_implements_it():
    """Same change, same reason, for the other unbuilt mech (set_env).

    The original intent — "a save changes intent; it never yanks a running
    model" — is preserved and then some: refusing cannot restart anything.
    """
    engine = FakeRestartEngine()

    with pytest.raises(EngineError, match="env\\+restart"):
        apply_settings("env+restart", engine_client=engine,
                       resolved={"KEY": {"value": "v"}})

    assert engine.env is None
    assert engine.restarted is False


def test_none_mech_applies_nothing_and_says_so():
    """A third-party endpoint you don't own is read-and-warn, permanently.
    Pretending otherwise would be a lie the UI then repeats."""
    result = apply_settings("none", engine_client=None, resolved={"a": {"value": "1"}})

    assert result["applied"] is False
    assert "cannot be configured" in result["reason"]


def test_node_settings_ships_the_document_and_requires_reload():
    """The node-settings mech (Plan C2): ship a full settings document to
    the node-agent and report requires_reload — no swap, no local write, an
    honest 'shipped, not yet live' outcome."""
    class FakeNode:
        sent = None

        def put_settings(self, profile, document):
            self.sent = (profile, document)

    node = FakeNode()
    result = apply_settings(
        "node-settings", engine_client=node,
        resolved={"max-model-len": {"value": "131072", "origin": "declared",
                                    "layer": "engine_model"}},
        profile="heretic", env={"V": "1"},
        argv=["serve", "/model", "--max-model-len", "131072"],
        service="aeon-vllm")

    assert node.sent == ("heretic", {
        "args": {"max-model-len": "131072"}, "env": {"V": "1"},
        "argv": ["serve", "/model", "--max-model-len", "131072"],
        "service": "aeon-vllm"})
    assert result == {"applied": True, "requires_reload": True,
                      "reason": "settings shipped to the node; reload to apply"}


def test_node_settings_requires_a_profile():
    """There is no per-node scope to ship to without one — unlike api/
    env+restart, which apply to whatever engine_client already points at."""
    with pytest.raises(ValueError):
        apply_settings("node-settings", engine_client=object(), resolved={})


def test_node_settings_does_not_swap():
    """Shipping settings must never trigger a multi-minute swap by itself —
    that stays the human's explicit Reload click (POST /api/spark/reload)."""
    class FakeNode:
        swapped = False

        def put_settings(self, profile, document):
            pass

        def swap(self, profile, force=False):
            self.swapped = True

    node = FakeNode()
    apply_settings("node-settings", engine_client=node,
                   resolved={}, profile="heretic", service="s")

    assert node.swapped is False


def test_unknown_mech_raises():
    with pytest.raises(ValueError):
        apply_settings("telepathy", engine_client=None, resolved={})


def test_empty_settings_is_a_no_op_not_a_wipe():
    """Applying nothing must not clear an engine's existing config."""
    engine = FakeApiEngine()

    result = apply_settings("api", engine_client=engine, resolved={})

    assert engine.configured is None
    assert result["applied"] is False


def test_unbuilt_mech_refusal_is_the_dedicated_subtype():
    """UnbuiltMechError subclasses EngineError so every non-HTTP
    ``except EngineError`` treatment is unchanged; the HTTP layer alone
    distinguishes it (501, see test_api's handler test)."""
    from app.configure import UnbuiltMechError

    with pytest.raises(UnbuiltMechError):
        apply_settings("api", engine_client=FakeApiEngine(),
                       resolved={"a": {"value": "1"}})
    assert issubclass(UnbuiltMechError, EngineError)
