"""Tests for app.configure — how settings reach an engine.

The write boundary is what each engine DECLARES it can do, not local vs
remote. That is what makes 'the Deck controls any connected inference
source' a rule rather than a pile of special cases — and it keeps an honest
slot for sources that genuinely cannot be written.
"""

import pytest

from app.configure import MECHS, apply_settings


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


def test_api_mech_applies_live_without_a_reload():
    engine = FakeApiEngine()

    result = apply_settings("api", engine_client=engine, resolved={"a": {"value": "1"}})

    assert engine.configured == {"a": "1"}
    assert result == {"applied": True, "requires_reload": False, "reason": "applied live"}


def test_env_restart_mech_sets_env_but_does_not_restart():
    """A save changes intent; it never yanks a running model. The restart
    is the human's explicit Reload."""
    engine = FakeRestartEngine()

    result = apply_settings("env+restart", engine_client=engine,
                            resolved={"KEY": {"value": "v"}})

    assert engine.env == {"KEY": "v"}
    assert engine.restarted is False
    assert result["requires_reload"] is True


def test_none_mech_applies_nothing_and_says_so():
    """A third-party endpoint you don't own is read-and-warn, permanently.
    Pretending otherwise would be a lie the UI then repeats."""
    result = apply_settings("none", engine_client=None, resolved={"a": {"value": "1"}})

    assert result["applied"] is False
    assert "cannot be configured" in result["reason"]


def test_node_settings_mech_is_explicitly_not_implemented_yet():
    """Better a clear NotImplementedError than a stub that reports success
    and silently drops the settings."""
    with pytest.raises(NotImplementedError, match="C2"):
        apply_settings("node-settings", engine_client=None, resolved={})


def test_unknown_mech_raises():
    with pytest.raises(ValueError):
        apply_settings("telepathy", engine_client=None, resolved={})


def test_empty_settings_is_a_no_op_not_a_wipe():
    """Applying nothing must not clear an engine's existing config."""
    engine = FakeApiEngine()

    result = apply_settings("api", engine_client=engine, resolved={})

    assert engine.configured is None
    assert result["applied"] is False
