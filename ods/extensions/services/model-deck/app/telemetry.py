"""Local GPU telemetry: a cached pass-through of dashboard-api's OWN
``GET /api/gpu/detailed`` (extensions/services/dashboard-api/gpu.py's
``IndividualGPU`` rows — the same numbers the GPU Monitor page already
shows for this box).

Attached to the LOCAL entry of ``/api/state``'s ``nodes[]`` block (see
``app.routers.status._nodes_block``) so the board reads ONE telemetry shape
for every node: a node-agent entry gets its ``gpus`` from
``app/node_observer.py``'s probe of that remote box, the local entry gets it
from here. Neither this module nor ``node_observer`` reads a GPU sysfs
itself for this purpose — dashboard-api already owns AMD/NVIDIA detection
and per-service assignment, and re-deriving that here would be a second,
inevitably-diverging copy of it.

Not a VERBATIM pass-through in one respect: the rows are filtered and
re-numbered into ``world.gpus``' own index vocabulary before they go out
(see ``_qualified``), because the board joins the two lists on that number.

TTL-cached (5 s — the ``app.node_clients.RemoteObserver`` idiom): a browser
tab polling ``/api/state`` costs dashboard-api nothing extra inside the
window, and a FAILED fetch is cached too, not just a successful one — a
dead dashboard-api must not be hammered on every poll.
"""

from __future__ import annotations

import time

import httpx

from app.gpu import MIN_VRAM_BYTES

# Every key from dashboard-api's IndividualGPU that the deck re-serves.
_ALLOWED = ("index", "uuid", "name", "memory_used_mb", "memory_total_mb",
            "memory_percent", "utilization_percent", "temperature_c", "power_w")
# assigned_services deliberately NOT carried: it is the install-time
# GPU_ASSIGNMENT_JSON_B64 config, not an observation (dashboard-api gpu.py:560),
# and the deck's own world observation is the engine-attribution authority.

_TIMEOUT = httpx.Timeout(5.0)


def _qualified(rows: list) -> list[dict]:
    """Allowed keys only, dropped below the deck's own VRAM bar, and
    RE-SEQUENCED — i.e. the same index vocabulary ``world.gpus`` speaks.

    dashboard-api enumerates every card it can see, integrated GPUs
    included (live autarch: a 2048 MB 0x13c0 display GPU sits alongside the
    two R9700s). ``app.gpu.read_gpus`` excludes anything under
    ``MIN_VRAM_BYTES`` and numbers what survives by POSITION — app/gpu.py:33
    ("it does not consume a slot in the returned index sequence") and
    app/gpu.py:110-113's ``"index": len(gpus)``.

    The two lists MUST agree, because the board joins them on that number:
    ``ui/src/model/nodes.ts``'s ``statsOf`` looks a card's stats up by the
    index its ``world.gpus`` capacity came from. A raw pass-through would
    silently hand GPU 0 the readings of whatever card enumerated first —
    correct today (the iGPU happens to enumerate last), wrong the moment it
    does not, and wrong with no error anywhere.

    ``MIN_VRAM_BYTES`` is IMPORTED, never re-typed: two copies of a
    qualifying bar is how the two lists come apart again later. ``uuid`` is
    carried through untouched — it is the identity a future join could
    verify against, which this positional one cannot.

    A row with no readable ``memory_total_mb`` cannot be judged against the
    bar and is dropped, exactly as ``read_gpus`` skips a card whose total it
    cannot read.
    """
    out: list[dict] = []
    for row in rows:
        kept = {k: row[k] for k in _ALLOWED if k in row}
        total_mb = kept.get("memory_total_mb")
        if total_mb is None or total_mb * 1024 * 1024 < MIN_VRAM_BYTES:
            continue
        kept["index"] = len(out)
        out.append(kept)
    return out


class LocalTelemetry:
    """``gpus()`` -> allowed-keys-only rows from dashboard-api, or ``None``
    on any fetch/parse failure or an unconfigured URL."""

    def __init__(self, settings, *, clock=time.monotonic, client=None) -> None:
        self._url = settings.dashboard_api_url
        self._headers = ({"Authorization": f"Bearer {settings.dashboard_api_key}"}
                         if settings.dashboard_api_key else {})
        self._clock = clock
        # Real client built lazily-but-once here, not per call — mirrors
        # every other engine client's constructor (e.g.
        # app.engines.comfyui.ComfyClient). None when unconfigured: nothing
        # should ever construct an httpx.Client against an empty base_url.
        self._client = client if client is not None else (
            httpx.Client(base_url=self._url, timeout=_TIMEOUT) if self._url else None)
        self._ttl_s = 5.0
        self._cached: list[dict] | None = None
        self._fetched_at: float | None = None

    def gpus(self) -> list[dict] | None:
        if not self._url:
            return None
        now = self._clock()
        if self._fetched_at is not None and now < self._fetched_at + self._ttl_s:
            return self._cached
        try:
            resp = self._client.get("/api/gpu/detailed", headers=self._headers)
            resp.raise_for_status()
            rows = resp.json()["gpus"]
            if not isinstance(rows, list):
                raise TypeError("gpus is not a list")
            result = _qualified(rows)
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            # Narrow I/O-boundary catch (repo CLAUDE.md): transport failure,
            # a non-list "gpus", a missing "gpus" key, or a non-JSON body
            # all degrade to the same "no telemetry right now" answer — the
            # board already renders null gracefully, and this is a
            # best-effort observation, not a control path.
            result = None
        # A failed fetch is cached too (see module docstring) — the TTL
        # clock is what re-tries, not this branch.
        self._cached = result
        self._fetched_at = now
        return result
