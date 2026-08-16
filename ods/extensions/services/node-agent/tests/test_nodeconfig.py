"""Config parsing must not crash-loop the container on a typo."""
import importlib
import socket

import pytest

import nodeconfig


def test_numeric_env_falls_back_to_default_when_unset(monkeypatch):
    monkeypatch.delenv("NODE_GPU_CACHE_TTL", raising=False)
    assert nodeconfig._env_float("NODE_GPU_CACHE_TTL", 2.0) == 2.0
    monkeypatch.setenv("NODE_GPU_CACHE_TTL", "   ")
    assert nodeconfig._env_float("NODE_GPU_CACHE_TTL", 2.0) == 2.0


def test_numeric_env_parses_a_value(monkeypatch):
    monkeypatch.setenv("NODE_GPU_CACHE_TTL", "0.5")
    assert nodeconfig._env_float("NODE_GPU_CACHE_TTL", 2.0) == 0.5


def test_numeric_env_fails_legibly(monkeypatch):
    """A bad value used to raise ValueError at import time, so the container
    crash-looped printing a raw traceback with no hint at the culprit."""
    monkeypatch.setenv("NODE_GPU_CACHE_TTL", "two seconds")
    with pytest.raises(SystemExit) as excinfo:
        nodeconfig._env_float("NODE_GPU_CACHE_TTL", 2.0)
    message = str(excinfo.value)
    assert "NODE_GPU_CACHE_TTL" in message
    assert "two seconds" in message


def test_string_env_treats_empty_as_absent(monkeypatch):
    """Compose's `${VAR:-}` idiom ships EMPTY, not absent — the class behind
    the 08-12 serving-blindness incident. The deck closed it with
    env_ignore_empty=True (model-deck app/settings.py:25); this is the
    agent-side equivalent: blank or whitespace-only means unset."""
    monkeypatch.setenv("NODE_NAME", "")
    assert nodeconfig._env_str("NODE_NAME", "fallback-host") == "fallback-host"
    monkeypatch.setenv("NODE_NAME", "   ")
    assert nodeconfig._env_str("NODE_NAME", "fallback-host") == "fallback-host"


def test_string_env_passes_a_real_value_stripped(monkeypatch):
    monkeypatch.setenv("GPU_BACKEND", " amd ")
    assert nodeconfig._env_str("GPU_BACKEND", "nvidia") == "amd"


def test_string_env_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("GPU_BACKEND", raising=False)
    assert nodeconfig._env_str("GPU_BACKEND", "nvidia") == "nvidia"


def test_blank_env_falls_back_at_import(monkeypatch):
    """Pins the WIRING, not just the helper: constants with real defaults
    (NODE_NAME -> hostname, GPU_BACKEND -> nvidia) must survive a compose
    file baking their env vars in empty. A helper nothing calls would pass
    the tests above while the module stays broken."""
    monkeypatch.setenv("NODE_NAME", "")
    monkeypatch.setenv("GPU_BACKEND", "")
    try:
        mod = importlib.reload(nodeconfig)
        assert mod.NODE_NAME == socket.gethostname()
        assert mod.GPU_BACKEND == "nvidia"
    finally:
        monkeypatch.undo()
        importlib.reload(nodeconfig)


def test_dead_port_var_is_not_reintroduced():
    """The port is owned by the Dockerfile CMD (``uvicorn --port``); a second
    copy in Python was never read and only added a crash-at-import path."""
    assert not hasattr(nodeconfig, "NODE_AGENT_PORT")
