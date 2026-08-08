"""Container-image provenance.

Two halves that nothing else in the deck has kept apart:
``version`` is the resolved content id (``sha256:…``) — exact, comparable,
never typed by a human — and ``label`` is the reference a human recognises
(``ds4-spark:v0.5.6``). app/engines/docker_ctl.py:105-123 has been carrying
a docstring asking for exactly this split ("an opaque identity for change
detection, not a human-readable version string"); this is where it lands.

Container inspect gives both from ONE read: top-level ``Image`` is the
content id, ``Config.Image`` is the reference the container was created
with. Neither needs a new socket-proxy permission — it is the same
``GET /containers/{name}/json`` running() already makes.
"""

from app import origins

_LOCALHOST = "localhost"


def origin_from_reference(reference: str) -> dict:
    """Parse ``[registry/]repository[:tag|@digest]`` into an origin record.

    Docker's registry rule, implemented exactly rather than approximated:
    the segment before the first ``/`` is a registry ONLY if it contains a
    ``.`` or a ``:`` or is literally ``localhost``. Otherwise it is a Docker
    Hub namespace and belongs to the repository — ``osmantic/ds4`` is one
    repository, not host ``osmantic``.
    """
    ref = (reference or "").strip()
    if not ref:
        raise ValueError("image reference must not be empty")

    head, slash, tail = ref.partition("/")
    if slash and ("." in head or ":" in head or head == _LOCALHOST):
        registry, remainder = head, tail
    else:
        registry, remainder = None, ref

    # Strip the tag or digest pin. A digest wins: "name@sha256:…" has a
    # colon inside the digest, so rsplit(":") alone would mangle it.
    name = remainder.split("@", 1)[0]
    if ":" in name.rsplit("/", 1)[-1]:
        name = name.rsplit(":", 1)[0]

    return {"registry": registry, "repository": name,
            "reference": ref, "build": None, "archive": None}


def identity_from_inspect(body: dict) -> dict:
    """Container inspect body -> identity. See the module docstring for why
    both fields come from this one read."""
    config = body.get("Config") or {}
    return {"version": body.get("Image"),
            "label": config.get("Image"),
            "detail": {}}


def identity_from_compose(reference: str | None, digest: str | None) -> dict:
    """A compose ``image:`` line, plus a digest IF one was proven to belong
    to this artifact. ``digest=None`` is the honest state for a sparky
    profile whose harvested catalog belongs to some other profile — see
    app.provenance_collect.spark_oci_entries."""
    return {"version": digest, "label": reference, "detail": {}}


def grade(identity: dict) -> str:
    """EXACT once a content id is in hand; otherwise nothing was measured."""
    return origins.EXACT if identity.get("version") else origins.UNKNOWN
