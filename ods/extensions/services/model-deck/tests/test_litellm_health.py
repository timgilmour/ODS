"""Tests for LiteLLM health interpretation.

The rule that matters: 'model does not exist' from a reachable node means
NOT-LOADED (normal on a single-slot engine), while a connection error means
the node is DOWN. On 2026-08-03, five of six 'unhealthy' routes were the
former and entirely correct.
"""

import httpx
import pytest

from app.engines import EngineError
from app.engines.litellm import LiteLLMClient, interpret_health


def _client(handler):
    return LiteLLMClient("http://litellm:4000", "k", transport=httpx.MockTransport(handler))


def test_health_returns_body():
    body = {"healthy_endpoints": [], "unhealthy_endpoints": []}
    client = _client(lambda request: httpx.Response(200, json=body))

    assert client.health() == body


def test_health_raises_engine_error_on_transport_failure():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(EngineError):
        _client(handler).health()


def test_healthy_endpoint_is_reachable_and_loaded():
    body = {
        "healthy_endpoints": [{"model": "spark-heretic"}],
        "unhealthy_endpoints": [],
    }

    result = interpret_health(body)

    assert result["spark-heretic"] == {
        "reachable": True, "loaded": True, "detail": "healthy",
    }


def test_model_not_found_is_reachable_but_not_loaded():
    """The single-slot normal case — NOT an alarm."""
    body = {
        "healthy_endpoints": [],
        "unhealthy_endpoints": [
            {"model": "spark-laguna",
             "error": "litellm.NotFoundError: The model `laguna` does not exist."}
        ],
    }

    result = interpret_health(body)

    assert result["spark-laguna"]["reachable"] is True
    assert result["spark-laguna"]["loaded"] is False


def test_connection_error_is_unreachable():
    body = {
        "healthy_endpoints": [],
        "unhealthy_endpoints": [
            {"model": "hipfire",
             "error": "APIConnectionError: Connection error."}
        ],
    }

    result = interpret_health(body)

    assert result["hipfire"]["reachable"] is False
    assert result["hipfire"]["loaded"] is False


def test_unrecognised_error_is_treated_as_unreachable():
    """Fail loud, not quiet: an error we don't have a rule for must not be
    silently downgraded to the benign not-loaded reading."""
    body = {
        "healthy_endpoints": [],
        "unhealthy_endpoints": [{"model": "x", "error": "kaboom"}],
    }

    result = interpret_health(body)

    assert result["x"]["reachable"] is False


def test_missing_error_field_does_not_raise():
    body = {"healthy_endpoints": [], "unhealthy_endpoints": [{"model": "x"}]}

    assert interpret_health(body)["x"]["reachable"] is False


def test_empty_body_yields_empty_mapping():
    assert interpret_health({}) == {}
