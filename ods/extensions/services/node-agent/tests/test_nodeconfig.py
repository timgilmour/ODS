"""Config parsing must not crash-loop the container on a typo."""
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


def test_dead_port_var_is_not_reintroduced():
    """The port is owned by the Dockerfile CMD (``uvicorn --port``); a second
    copy in Python was never read and only added a crash-at-import path."""
    assert not hasattr(nodeconfig, "NODE_AGENT_PORT")
