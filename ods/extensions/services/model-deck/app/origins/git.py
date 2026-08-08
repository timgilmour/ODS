"""Source-repo provenance — including ComfyUI node packs, which are just
git checkouts (design decision D7).

``diverges_from_default`` is the ds4 trap made into a field. ds4's release
tags live on ``batched-serving``, 569 ahead / 216 behind ``main``, so
"is there a newer version" is NOT a tag comparison and any future
update-checker must be able to see that before it suggests anything.

v1 records; it does not verify (D10). The deck container has no path to
``~/ds4`` or ``~/projects/*`` and there is no host-agent endpoint that runs
``git rev-parse``. ``grade`` takes a ``run_git`` so the seam is real and
named, and production passes ``None`` — which reports UNAVAILABLE, the
state that means "could not check", never UNKNOWN, which would claim we
never tried.
"""

from app import origins

REF_KINDS = ("tag", "branch", "commit")


def origin(remote: str, ref: str, ref_kind: str,
           default_branch: str | None = None,
           diverges_from_default: bool | None = None) -> dict:
    if ref_kind not in REF_KINDS:
        raise ValueError(
            f"ref_kind must be one of {list(REF_KINDS)}, got {ref_kind!r}")
    return {"remote": remote, "ref": ref, "ref_kind": ref_kind,
            "default_branch": default_branch,
            "diverges_from_default": diverges_from_default}


def identity(commit: str | None, ref: str | None) -> dict:
    return {"version": commit, "label": ref, "detail": {}}


def grade(identity: dict, run_git=None) -> str:
    """UNAVAILABLE without a git runner — "could not check", not "unknown"."""
    if run_git is None:
        return origins.UNAVAILABLE
    return origins.EXACT if identity.get("version") else origins.UNKNOWN
