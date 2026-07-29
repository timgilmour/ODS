"""Bearer-auth hardening for a service that binds all interfaces."""
from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def test_verify_key_uses_constant_time_compare(monkeypatch):
    """The bearer comparison must be constant-time.

    Both other ODS bearer checks (dashboard-api/security.py and
    ods-host-agent.py) use secrets.compare_digest; a plain ``!=`` short-circuits
    on the first differing byte and leaks key material through response timing
    on a network-exposed port.
    """
    calls = []
    real = app_module.secrets.compare_digest

    def spy(left, right):
        calls.append((left, right))
        return real(left, right)

    monkeypatch.setattr(app_module.secrets, "compare_digest", spy)
    assert client.get("/v1/node/gpu", headers=AUTH).status_code == 200
    assert calls, "verify_key must compare bearer tokens with compare_digest"


def test_non_ascii_authorization_header_is_a_clean_401():
    """compare_digest raises TypeError on non-ASCII ``str`` and the header is
    attacker-controlled, so the comparison must run over UTF-8 bytes or an
    unauthenticated request becomes a 500."""
    # Sent as raw bytes: httpx refuses a non-ASCII header *str*, but a real
    # client can put any bytes on the wire, and Starlette decodes them to a
    # non-ASCII str before the dependency sees them.
    r = client.get("/v1/node/gpu",
                   headers={"Authorization": "Bearer café".encode("utf-8")})
    assert r.status_code == 401
    assert r.content == b""


def test_wrong_key_prefix_is_rejected():
    for header in ("test-key", "Basic test-key", "Bearer  test-key",
                   "Bearer test-key-extra", "bearer test-key"):
        r = client.get("/v1/node/gpu", headers={"Authorization": header})
        assert r.status_code == 401, header
        assert r.content == b""


def test_openapi_surface_is_disabled():
    """The docs/schema routes are unauthenticated by construction, and this
    service binds all interfaces via network_mode: host, so they must not
    advertise the API to anyone who can reach the port."""
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path
