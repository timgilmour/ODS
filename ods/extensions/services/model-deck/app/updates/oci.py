"""Registry checks.

TWO KINDS BECAUSE REGISTRIES CARRY TWO KINDS OF TAG. A version tag names a
release and the question is "is there a higher one" (check_tags). A CHANNEL
tag -- `slim`, `latest`, `stable` -- names a moving pointer and the question
is "has mine moved" (check_channel), which is answered exactly by comparing
digests and needs no ranking at all.

comfyui-aeon-spark is the live proof the distinction is real: its six tags are
`slim`/`full`/`latest` and friends. Ranking them is meaningless; comparing the
digest behind `:slim` against the pinned one is exact.
"""

from app import updates
from app.updates import ordering

_DEFAULT_REGISTRY = "ghcr.io"
_MANIFEST_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def _result(source, status, *, current=None, latest=None, detail=None, note=None):
    return {"id": source["id"], "status": status, "current": current,
            "latest": latest, "detail": detail or {}, "note": note}


def _token(registry: str, repository: str, fetch) -> str | None:
    """Anonymous pull token. ghcr.io answers /token for public repos; a
    registry that does not simply yields no token and the request proceeds
    unauthenticated."""
    response = fetch(
        f"https://{registry}/token?scope=repository:{repository}:pull")
    if response["status_code"] != 200:
        return None
    body = response["json"]
    # `json` is typed dict | list | None; a token endpoint answering with
    # anything but a dict is a malformed read, not a token -- treat it the
    # same as "no token" rather than crashing on `.get`.
    return body.get("token") if isinstance(body, dict) else None


def _auth(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def check_channel(source: dict, fetch) -> dict:
    registry = source.get("registry") or _DEFAULT_REGISTRY
    repository = source["repository"]
    reference = source["reference"]
    pinned = source["pinned"]

    token = _token(registry, repository, fetch)
    headers = {**_auth(token), "Accept": _MANIFEST_ACCEPT}
    response = fetch(
        f"https://{registry}/v2/{repository}/manifests/{reference}",
        headers=headers, method="HEAD")
    if response["status_code"] != 200:
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note=f"registry returned {response['status_code']}")

    digest = (response["headers"] or {}).get("Docker-Content-Digest")
    if not digest:
        # Read failure, NOT evidence the tag moved. Claiming `available` here
        # would invent an upgrade out of a missing header.
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note="registry returned no Docker-Content-Digest header")

    status = updates.CURRENT if digest == pinned else updates.AVAILABLE
    return _result(source, status, current=pinned, latest=digest,
                   detail={"reference": reference})


def check_tags(source: dict, fetch) -> dict:
    registry = source.get("registry") or _DEFAULT_REGISTRY
    repository = source["repository"]
    pinned = source["pinned"]

    token = _token(registry, repository, fetch)
    response = fetch(f"https://{registry}/v2/{repository}/tags/list",
                     headers=_auth(token))
    if response["status_code"] != 200:
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note=f"registry returned {response['status_code']}")

    body = response["json"]
    # Same defensive note as `_token`: `json` may legally be a list here per
    # the fetch contract. A tags/list body that isn't a dict is a malformed
    # read -- fold it into "no tags" rather than crashing on `.get`.
    tags = (body.get("tags") or []) if isinstance(body, dict) else []
    if not tags:
        # No tags at all is a failure to LEARN anything about the repository,
        # not evidence of being current -- distinct from tags-present-but-
        # unparseable (UNDETERMINED, below). `rank([], ...)` would happily
        # hand back `newer: [], rankable: False`, which the naive mapping
        # reads as "nothing newer" -- a false CURRENT built on having read
        # nothing. Checked before ranking so ordering.rank is never asked to
        # judge an empty set.
        return _result(source, updates.UNAVAILABLE, current=pinned,
                       note="registry returned no tags")

    ranked = ordering.rank(tags, source["order"], pinned)
    detail = {"newer": ranked["newer"], "unranked": ranked["unranked"],
              "tag_count": len(tags)}

    if not ranked["rankable"]:
        status = updates.UNDETERMINED if ranked["newer"] else updates.CURRENT
        return _result(source, status, current=pinned, latest=ranked["latest"],
                       detail=detail)

    status = updates.AVAILABLE if ranked["newer"] else updates.CURRENT
    return _result(source, status, current=pinned, latest=ranked["latest"],
                   detail=detail)
