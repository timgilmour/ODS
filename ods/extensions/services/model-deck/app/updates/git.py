"""Git checks over the GitHub HTTP API.

NOT `git ls-remote`: the deck image carries no git binary (verified
in-container 2026-08-08). The compare endpoint is also strictly better for
this purpose -- it returns ahead_by/behind_by directly, which ls-remote
cannot.

DIVERGENCE IS CONTEXT, NOT A VERDICT. ds4's tags live on `batched-serving`,
569 ahead / 216 behind `main`. There is no status word for which "diverged"
is a sensible answer, so it never becomes one: the verdict comes from the tag
comparison and the divergence rides along in `detail`, which is exactly the
trap the provenance design wanted visible before anyone acts on a suggestion.

Sibling of `app.updates.oci`, solving the same problem for registries -- same
shape, same defensive posture. `fetch`'s `json` is typed dict | list | None
throughout this module (GitHub's tags endpoint answers with a list; its error
bodies are dicts), so every read guards the shape it expects before touching
it rather than assuming the happy path.
"""

import re

from app import updates
from app.updates import ordering

_API = "https://api.github.com"
_REMOTE_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")
_MALFORMED_TAGS_NOTE = "github returned a malformed tags response"
_NO_TAGS_NOTE = "github returned no tags"
_MALFORMED_COMPARE_NOTE = "github returned a malformed compare response"


def parse_remote(remote: str) -> tuple[str, str] | None:
    """(owner, repo) for a GitHub remote; None for anything else."""
    m = _REMOTE_RE.match((remote or "").strip())
    return (m[1], m[2]) if m else None


def _result(source, status, *, current=None, latest=None, detail=None, note=None):
    return {"id": source["id"], "status": status, "current": current,
            "latest": latest, "detail": detail or {}, "note": note}


def _unsupported(source):
    return _result(source, updates.UNAVAILABLE, current=source.get("pinned"),
                   note="remote is not on github.com; no checker available")


def _failure(source, response):
    """Classify a non-200. Rate limiting is expected operation, not a defect
    -- but only a CONFIRMED `X-RateLimit-Remaining: 0` earns that note. A
    bare 403 (missing header, or present with a nonzero value) is an
    ordinary failure and must not be conflated with rate limiting."""
    code = response["status_code"]
    headers = response["headers"] or {}
    if code == 403 and headers.get("X-RateLimit-Remaining") == "0":
        return _result(source, updates.UNAVAILABLE, current=source.get("pinned"),
                       note="github rate limit exhausted")
    if code in (301, 302, 307, 308):
        # THE REMEDY NAMES THE RIGHT ROUTE. For a git check the remote lives
        # on the WATCH SOURCE (`source["remote"]`, read at :101/:135), not on
        # the artifact's `origin`, so PUT /api/provenance/origin fixes
        # nothing here -- PUT /api/provenance/watch does. This is the exact
        # live case the design cites (Kaden-Schutt/hipfire -> warpfront/
        # hipfire), and an operator following the old wording would edit the
        # one field the checker never reads.
        return _result(source, updates.UNAVAILABLE, current=source.get("pinned"),
                       detail={"moved": True, "location": headers.get("Location")},
                       note="repository has moved; re-declare the watch "
                            "source's remote (PUT /api/provenance/watch)")
    return _result(source, updates.UNAVAILABLE, current=source.get("pinned"),
                   note=f"github returned {code}")


def _compare(owner, repo, base, head, fetch):
    return fetch(f"{_API}/repos/{owner}/{repo}/compare/{base}...{head}")


def _is_int(value) -> bool:
    """True for a genuine int, excluding bool (a `bool` is an `int`
    subclass in Python, but a JSON `true`/`false` is not a count)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _latest_sha(body: dict) -> str | None:
    """The most recent commit sha from a compare body's `commits` list,
    defensively. `commits` may legitimately be absent, empty, not a list,
    or contain non-dict elements (or a dict with a non-string `sha`) --
    none of that may crash the check, it should just fail to produce a
    `latest`. Unlike `ahead_by`/`behind_by` this never drives the verdict,
    so defaulting to None on anything unreadable is safe here."""
    commits = body.get("commits")
    if isinstance(commits, list) and commits:
        last = commits[-1]
        if isinstance(last, dict) and isinstance(last.get("sha"), str):
            return last["sha"]
    return None


def check_compare(source: dict, fetch) -> dict:
    parsed = parse_remote(source.get("remote", ""))
    if parsed is None:
        return _unsupported(source)
    owner, repo = parsed
    pinned = source["pinned"]

    response = _compare(owner, repo, pinned, source["ref"], fetch)
    if response["status_code"] != 200:
        return _failure(source, response)

    body = response["json"]
    if not isinstance(body, dict):
        # A 200 whose body isn't an object cannot be read as a compare
        # result at all -- a failure to PARSE, not "identical".
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note=_MALFORMED_COMPARE_NOTE)

    ahead_by, behind_by = body.get("ahead_by"), body.get("behind_by")
    if not _is_int(ahead_by) or not _is_int(behind_by):
        # `behind_by` drives the verdict directly. Defaulting an unreadable
        # value to 0 would manufacture CURRENT out of a field we could not
        # read -- exactly what this module exists to refuse. Fail loudly
        # instead, the same discipline `_divergence` already applies to its
        # own (context-only) reading of these two fields.
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note=_MALFORMED_COMPARE_NOTE)

    status = updates.AVAILABLE if behind_by > 0 else updates.CURRENT
    return _result(source, status, current=pinned, latest=_latest_sha(body),
                   detail={"ahead_by": ahead_by, "behind_by": behind_by,
                           "compare_status": body.get("status")})


def check_tags(source: dict, fetch) -> dict:
    parsed = parse_remote(source.get("remote", ""))
    if parsed is None:
        return _unsupported(source)
    owner, repo = parsed
    pinned = source["pinned"]

    response = fetch(f"{_API}/repos/{owner}/{repo}/tags?per_page=100")
    if response["status_code"] != 200:
        return _failure(source, response)

    raw = response["json"]
    if raw is None or raw == []:
        # Genuinely nothing there: absent body or an empty list. A failure
        # to LEARN anything about the repository, not evidence of being
        # current -- distinct from tags-present-but-unparseable
        # (malformed, below). `ordering.rank([], ...)` would happily hand
        # back `newer: [], rankable: False`, which the naive mapping reads
        # as "nothing newer" -- a false CURRENT built on having read
        # nothing. Checked before ranking so ordering.rank is never asked
        # to judge an empty set.
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note=_NO_TAGS_NOTE)

    if not isinstance(raw, list):
        # GitHub's tags endpoint answers with a list; its error bodies are
        # dicts (e.g. {"message": "Not Found"}). Iterating a dict yields its
        # KEYS -- strings -- and calling `.get` on a string crashes. Fail
        # the read instead of guessing what was meant.
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note=_MALFORMED_TAGS_NOTE)

    tags = [t.get("name") for t in raw
            if isinstance(t, dict) and isinstance(t.get("name"), str)]
    if not tags:
        # Elements were present but none yielded a usable string name --
        # same "could not read it" category as a non-list body, not "no
        # tags" (something WAS there, just not usable).
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note=_MALFORMED_TAGS_NOTE)

    ranked = ordering.rank(tags, source["order"], pinned)
    detail = {"newer": ranked["newer"], "unranked": ranked["unranked"],
              "tag_count": len(tags)}

    divergence = _divergence(source, owner, repo, fetch)
    if divergence is not None:
        detail["diverges_from_default"] = divergence

    if not ranked["rankable"]:
        status = updates.UNDETERMINED if ranked["newer"] else updates.CURRENT
    else:
        status = updates.AVAILABLE if ranked["newer"] else updates.CURRENT
    return _result(source, status, current=pinned, latest=ranked["latest"],
                   detail=detail)


def _divergence(source, owner, repo, fetch) -> dict | None:
    """ahead/behind of the tag-bearing ref vs the default branch, when the
    source declares both. Context only -- never changes the verdict, and any
    failure to measure it (bad status, malformed body, non-int fields) is
    silently omitted rather than downgrading a good tag answer or crashing.
    A partially-trustworthy body (e.g. one int field garbled) is also
    omitted whole rather than reported half-guessed."""
    default_branch = source.get("default_branch")
    ref = source.get("ref")
    if not default_branch or not ref or default_branch == ref:
        return None
    response = _compare(owner, repo, default_branch, ref, fetch)
    if response["status_code"] != 200:
        return None
    body = response["json"]
    if not isinstance(body, dict):
        return None
    ahead_by, behind_by = body.get("ahead_by"), body.get("behind_by")
    if not _is_int(ahead_by) or not _is_int(behind_by):
        return None
    return {"ahead_by": ahead_by, "behind_by": behind_by}
