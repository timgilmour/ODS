"""Ranking for tag-set checks — the ONLY place update-checking can be wrong
rather than merely unavailable.

Every other check in this package is exact: a digest either matches or it
does not, a commit either is an ancestor or it is not. Ranking is different,
because "newer" is a judgement that depends on a convention the registry does
not record. So the convention is DECLARED per watch source and this module
applies it literally.

A tag it cannot parse under the declared order is EXCLUDED from ranking and
returned in `unranked` -- never coerced into a comparison. The real tag sets
this was written against make the reason concrete: aeon-vllm-ultimate carries
`2026-07-27-v0.26.0` next to `v0.22.1-pr44389-spark` and `pre-audio`, and
comfyui-aeon-spark's tags are `slim`/`full`/`latest` -- channel names with no
ordering at all. A generic version sort over either produces a confident
answer that happens to be nonsense.
"""

import re

ORDERS = ("semver", "date", "none")

_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _semver_key(tag: str) -> tuple[int, int, int] | None:
    m = _SEMVER_RE.match(tag)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def _date_key(tag: str) -> tuple[int, int, int] | None:
    m = _DATE_RE.match(tag)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


_KEYS = {"semver": _semver_key, "date": _date_key}


def rank(tags: list[str], order: str, pinned: str) -> dict:
    """Rank `tags` under `order` relative to `pinned`.

    Returns `latest` (None when unrankable), `newer` (strictly newer than the
    pin, or every non-pin tag under `none`), `unranked` (seen but unparseable)
    and `rankable`.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {list(ORDERS)}, got {order!r}")

    if order == "none":
        # Nothing is comparable, so nothing is claimed. Every tag that is not
        # the pin is reported as "new to you" and the operator ranks them.
        return {"latest": None, "newer": sorted(t for t in tags if t != pinned),
                "unranked": [], "rankable": False}

    key = _KEYS[order]
    ranked: list[tuple[tuple[int, int, int], str]] = []
    unranked: list[str] = []
    for tag in tags:
        k = key(tag)
        if k is None:
            unranked.append(tag)
        else:
            ranked.append((k, tag))

    if not ranked:
        return {"latest": None, "newer": [], "unranked": sorted(unranked),
                "rankable": False}

    ranked.sort()
    latest = ranked[-1][1]
    pin_key = key(pinned)
    if pin_key is None:
        # The pin itself is unparseable: report the newest tag as new rather
        # than pretending to compare against something we could not read.
        newer = [latest]
    else:
        newer = [tag for k, tag in ranked if k > pin_key]

    return {"latest": latest, "newer": sorted(newer),
            "unranked": sorted(unranked), "rankable": True}
