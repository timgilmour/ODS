"""The production HTTP seam.

THE ONLY PLACE THIS PACKAGE TOUCHES THE NETWORK. Every checker takes `fetch`
as a parameter, so the entire unit suite runs offline with a fake -- no
cassettes, no flakes, no live GitHub in CI. Same reasoning as the provenance
module's injected `run_git`.

A PER-PASS BUDGET, NOT A RATE LIMITER. Anonymous GitHub allows 60 requests an
hour per IP and that ceiling is shared with everything else on the box, so a
runaway pass must not be able to spend it. Exhausting the budget raises, and
the caller turns that into UNAVAILABLE for the affected source -- expected
operation, not a defect. The counter is incremented BEFORE the request is
attempted, so a request that then fails still spends its slot -- otherwise a
permanently broken upstream could be retried without limit every pass.

CASE-INSENSITIVE HEADERS, DELIBERATELY NOT MATERIALIZED INTO A PLAIN DICT.
`dict(a_real_response.headers)` lowercases every key: constructing
`httpx.Headers([("X-RateLimit-Remaining", "0")])` and converting it to a
dict yields `{"x-ratelimit-remaining": "0"}`, unconditionally, regardless of
the case the server actually sent (confirmed interactively; it holds even
though the raw wire bytes keep their original case in `Headers.raw`, because
`.items()`/`dict()` normalize on the way out). `app.updates.git._failure`
reads `headers.get("X-RateLimit-Remaining")` and `headers.get("Location")`;
`app.updates.oci.check_channel` reads `headers.get("Docker-Content-Digest")`
-- all exact-case lookups on a plain dict, and both modules are reviewed and
out of scope for this task. Flattening the response headers here would
silently disable rate-limit detection AND the origin-moved 301/Location read
this feature calls load-bearing (`Kaden-Schutt/hipfire` -> `warpfront/hipfire`)
-- and no existing test would catch it, because every git.py/oci.py test
injects its own hand-built fake fetch and never touches this module's
conversion at all. So the header mapping is returned AS RECEIVED from
`client.request` (a real `httpx.Headers` in production; whatever a test's
fake supplies) instead of being flattened. Confirmed safe to do: every
caller only ever calls `.get(one exact key)` on it -- never iterates,
serializes, or isinstance-checks it (it is never persisted; only individual
extracted string values are, e.g. into a result's `detail`).

THE ETAG CACHE STORES HEADERS TOO, not just the body. A 304 stands in for
the 200 it validates against, and that includes headers -- a real 304
typically omits validators like `Docker-Content-Digest` or `Location`
entirely, so replaying the FRESH 304's own near-empty headers instead of the
ORIGINAL 200's would quietly break any future header-reading checker on a
cache hit. No checker today is affected (`oci.check_channel` is a HEAD with
no body, so it never populates the cache -- see the `payload is not None`
guard below), but it costs nothing to get right now instead of later.

A CALLER-SUPPLIED `client` MUST ITSELF BE BUILT WITH `follow_redirects=False`.
This module does not, and -- to keep the injectable-fake contract every test
here relies on -- cannot, pass `follow_redirects` on the per-request call:
the fake `client.request(method, url, headers=None, timeout=None)` shape
this suite (and presumably Task 8's) uses has no such parameter. The default
client built when none is supplied already sets it at construction time.

THE ETAG CACHE IS UNBOUNDED BY DESIGN, NOT BY OVERSIGHT. It is keyed on URL
alone and lives for the process lifetime, but the thing it caches is a
human-edited watch list -- on the order of 7 sources today, each contributing
at most two URLs (a tags list, and for git_compare a compare URL too) -- not
attacker- or user-controlled data that could grow without bound. At that
scale a cap/LRU would add real eviction-bug surface for no measurable memory
benefit. Revisit only if watch sources become dynamically generated at a
much larger scale.

THE DEFAULT CLIENT IS NEVER EXPLICITLY CLOSED, AND THAT IS FINE HERE.
`httpx.Client` holds no reference cycle in its own implementation (confirmed:
disabling the cyclic GC and dropping the only reference still frees it
immediately), so CPython's ordinary refcounting closes its sockets as soon as
the `fetch` closure returned by `make_fetch` goes out of scope -- no `with`
block needed for correctness. Task 8's update-check thread should still
build ONE `client` and pass it into every pass's `make_fetch(client=...)`
call rather than accept a fresh default each pass, purely to reuse TCP/TLS
connections across the six-hour cadence -- an efficiency choice, not a
leak-safety one.

A CACHE KEYED ON URL ALONE CANNOT DISTINGUISH TWO REQUESTS TO THE SAME URL
MADE UNDER DIFFERENT CREDENTIALS. `oci._token` mints a fresh bearer token on
every single call, so the identical tags/manifest URL is fetched with a
DIFFERENT `Authorization` header on every pass, by design. Keying the cache
on (url, headers) instead would close that theoretical gap but would also
mean the cache never hits for oci.py's real traffic, defeating the whole
feature it exists for. Accepted as-is: every watch source here is a public,
anonymously-fetched artifact where the token is a formality, not an access
boundary, so a 304 genuinely does mean "the origin says this is unchanged"
regardless of which equivalently-scoped credential asked. This would need
revisiting before this ever watches a private repo or registry.
"""

import httpx


class FetchError(RuntimeError):
    """The request could not be made or completed."""


def make_fetch(*, timeout_s: float = 8.0, budget: int = 40,
               etags: dict | None = None, client=None):
    """Build a `fetch(url, *, headers=None, method="GET") -> dict`.

    `etags` persists across passes so an unchanged tag list costs a 304
    rather than a full body; it is caller-owned so its lifetime is explicit.

    `client` is the injectable seam every test in this module uses. Leave it
    unset in production and a real `httpx.Client(follow_redirects=False)` is
    built once for this closure's lifetime (see the module docstring for why
    never explicitly closing it is fine, and why a caller that wants to
    reuse connections across passes should build and pass its own).
    """
    etags = etags if etags is not None else {}
    client = client if client is not None else httpx.Client(follow_redirects=False)
    spent = {"n": 0}

    def fetch(url: str, *, headers: dict | None = None, method: str = "GET") -> dict:
        if spent["n"] >= budget:
            raise FetchError(f"request budget of {budget} exhausted")
        spent["n"] += 1

        request_headers = dict(headers or {})
        cached = etags.get(url)
        if cached:
            request_headers["If-None-Match"] = cached["etag"]

        try:
            response = client.request(method, url, headers=request_headers,
                                      timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 -- every transport failure is one word
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code == 304 and cached:
            return {"status_code": 200, "headers": cached["headers"],
                    "json": cached["json"]}

        try:
            payload = response.json()
        except Exception:  # noqa: BLE001 -- a non-JSON body is not an error here
            payload = None

        etag = response.headers.get("ETag")
        if etag and response.status_code == 200 and payload is not None:
            etags[url] = {"etag": etag, "json": payload,
                          "headers": response.headers}

        return {"status_code": response.status_code,
                "headers": response.headers, "json": payload}

    return fetch
