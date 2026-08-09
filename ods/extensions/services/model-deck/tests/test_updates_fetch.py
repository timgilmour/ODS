import httpx
import pytest

from app.updates import fetch as fetch_mod


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def request(self, method, url, headers=None, timeout=None):
        self.calls.append((method, url, headers))
        return self.responses.pop(0)


class FakeResponse:
    def __init__(self, status_code, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_budget_stops_further_requests():
    client = FakeClient([FakeResponse(200, payload={}) for _ in range(5)])
    fetch = fetch_mod.make_fetch(budget=2, client=client)
    fetch("https://a/1")
    fetch("https://a/2")
    with pytest.raises(fetch_mod.FetchError):
        fetch("https://a/3")
    assert len(client.calls) == 2


def test_a_transport_error_becomes_fetch_error():
    class Boom:
        def request(self, *args, **kwargs):
            raise OSError("no route to host")

    fetch = fetch_mod.make_fetch(client=Boom())
    with pytest.raises(fetch_mod.FetchError):
        fetch("https://a/1")


def test_a_body_that_is_not_json_yields_none_not_a_crash():
    client = FakeClient([FakeResponse(200, payload=None)])
    fetch = fetch_mod.make_fetch(client=client)
    assert fetch("https://a/1")["json"] is None


def test_an_etag_is_sent_and_a_304_replays_the_cached_body():
    client = FakeClient([
        FakeResponse(200, headers={"ETag": "W/\"abc\""}, payload={"tags": ["v1"]}),
        FakeResponse(304, headers={"ETag": "W/\"abc\""}),
    ])
    etags = {}
    fetch = fetch_mod.make_fetch(client=client, etags=etags)
    first = fetch("https://a/tags")
    second = fetch("https://a/tags")

    assert first["json"] == {"tags": ["v1"]}
    assert second["status_code"] == 200          # replayed, not surfaced as 304
    assert second["json"] == {"tags": ["v1"]}
    assert client.calls[1][2]["If-None-Match"] == "W/\"abc\""


# --- Risk coverage below. See app/updates/fetch.py's module docstring for
# the reasoning behind each of these; this file exists to prove it rather
# than just assert it in prose. ---


def test_headers_are_case_insensitively_readable_like_a_real_response():
    """`git._failure` reads `headers.get("X-RateLimit-Remaining")` and
    `headers.get("Location")`; `oci.check_channel` reads
    `headers.get("Docker-Content-Digest")` -- all exact case, and both
    modules are out of scope for this task. A real httpx response's headers
    lowercase every key on iteration/dict() regardless of wire case (proven
    interactively: `dict(httpx.Headers([("X-RateLimit-Remaining", "0")]))`
    yields `{"x-ratelimit-remaining": "0"}`). A plain-dict test double can't
    catch a regression here -- it never goes through that normalization --
    so this test uses a real `httpx.Headers` instance, matching what
    production actually hands back."""
    real_headers = httpx.Headers([
        ("X-RateLimit-Remaining", "0"),
        ("Location", "https://github.com/warpfront/hipfire"),
        ("Docker-Content-Digest", "sha256:deadbeef"),
    ])
    client = FakeClient([FakeResponse(403, headers=real_headers, payload=None)])
    fetch = fetch_mod.make_fetch(client=client)
    headers = fetch("https://a/1")["headers"]

    assert headers.get("X-RateLimit-Remaining") == "0"
    assert headers.get("Location") == "https://github.com/warpfront/hipfire"
    assert headers.get("Docker-Content-Digest") == "sha256:deadbeef"


def test_a_304_with_no_cached_entry_does_not_crash_or_fake_a_200():
    """A 304 can arrive for a URL this process has no cache entry for -- a
    fresh process restart, or an out-of-band ETag from something else
    entirely. It must surface honestly (304, no body) rather than being
    mistaken for the 200-replay path or blowing up trying to serve a cache
    entry that doesn't exist."""
    client = FakeClient([FakeResponse(304, headers={"ETag": "W/\"abc\""})])
    fetch = fetch_mod.make_fetch(client=client, etags={})
    result = fetch("https://a/tags")

    assert result["status_code"] == 304
    assert result["json"] is None


def test_a_304_with_no_cached_entry_does_not_poison_the_cache():
    """Following on from the above: a bare 304 with nothing cached must not
    itself become a cache entry (it has no body to serve later)."""
    client = FakeClient([FakeResponse(304, headers={"ETag": "W/\"abc\""})])
    etags = {}
    fetch = fetch_mod.make_fetch(client=client, etags=etags)
    fetch("https://a/tags")

    assert etags == {}


def test_budget_is_spent_by_a_request_that_then_fails():
    """The budget must count an attempt, not just a success -- otherwise a
    permanently broken upstream could be retried without limit every pass."""
    class Boom:
        def __init__(self):
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            raise OSError("no route to host")

    client = Boom()
    fetch = fetch_mod.make_fetch(budget=1, client=client)
    with pytest.raises(fetch_mod.FetchError):
        fetch("https://a/1")
    with pytest.raises(fetch_mod.FetchError):
        fetch("https://a/2")

    assert client.calls == 1  # second call never reached the transport


def test_a_304_replay_returns_the_original_responses_headers():
    """The etag cache stands in for a whole prior response, headers
    included -- a bare 304 typically omits validators like
    Docker-Content-Digest or Location entirely, so replaying the FRESH
    304's near-empty headers instead of the ORIGINAL 200's would quietly
    break any future header-reading checker on a cache hit."""
    original_headers = {"ETag": "W/\"abc\"", "Docker-Content-Digest": "sha256:aaa"}
    client = FakeClient([
        FakeResponse(200, headers=original_headers, payload={"tags": ["v1"]}),
        FakeResponse(304, headers={"ETag": "W/\"abc\""}),  # a real 304: sparse
    ])
    fetch = fetch_mod.make_fetch(client=client, etags={})
    fetch("https://a/tags")
    second = fetch("https://a/tags")

    assert second["headers"].get("Docker-Content-Digest") == "sha256:aaa"


def test_a_cache_hit_does_not_distinguish_which_headers_asked():
    """DELIBERATE, documented behaviour, not an oversight: the cache is
    keyed on URL alone. `oci._token` mints a fresh bearer token every call,
    so the identical tags/manifest URL is fetched with a DIFFERENT
    Authorization header on every single pass by design -- keying the cache
    on (url, headers) would make it never hit for real OCI traffic, which
    defeats the feature. Accepted because every watch source here is a
    public, anonymously-fetched artifact where the token is a formality,
    not an access boundary. This test exists so a future change to this
    behaviour is a deliberate decision, not a silent regression."""
    client = FakeClient([
        FakeResponse(200, headers={"ETag": "W/\"abc\""}, payload={"tags": ["v1"]}),
        FakeResponse(304, headers={"ETag": "W/\"abc\""}),
    ])
    fetch = fetch_mod.make_fetch(client=client, etags={})
    fetch("https://a/tags", headers={"Authorization": "Bearer token-A"})
    second = fetch("https://a/tags", headers={"Authorization": "Bearer token-B"})

    assert second["status_code"] == 200
    assert second["json"] == {"tags": ["v1"]}


def test_default_client_is_built_with_follow_redirects_false(monkeypatch):
    """Untested branch in the reference implementation: none of the given
    tests ever omit `client=`, so the real default-construction path (the
    one production actually uses) was never exercised. `follow_redirects`
    must be False -- app.updates.git relies on seeing a bare 301, not a
    silently-followed one -- so this locks the default in without making a
    real network call (constructing an httpx.Client does no I/O by itself)."""
    captured = {}

    class SpyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(fetch_mod.httpx, "Client", SpyClient)
    fetch_mod.make_fetch()

    assert captured == {"follow_redirects": False}
