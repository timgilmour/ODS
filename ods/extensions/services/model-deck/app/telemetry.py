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

import logging
import time
from pathlib import Path

import httpx

from app.gpu import MIN_VRAM_BYTES

_log = logging.getLogger(__name__)

# Every key from dashboard-api's IndividualGPU that the deck re-serves.
#
# The three `*_available` booleans are NOT optional garnish: dashboard-api
# emits a required numeric PLUS its flag, and on a sensor it could not read it
# sends the value 0 with the flag False (dashboard-api/gpu.py:172 —
# `temperature_available=temp > 0`; node-agent/models.py:23-38 declares the
# same triple). Carrying the number without the flag turns "we failed to read
# this sensor" into a real 0 degC / 0% / 0 MB-used reading, which is exactly
# what the board would then meter. The fold back into a nullable reading
# happens once, UI-side, in ui/src/model/nodes.ts's `statsOf`/`gpuCapacity`.
_ALLOWED = ("index", "uuid", "name", "memory_used_mb", "memory_total_mb",
            "memory_percent", "utilization_percent", "temperature_c", "power_w",
            "memory_usage_available", "utilization_available", "temperature_available")
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
        self._key = settings.dashboard_api_key
        self._key_file = Path(settings.dashboard_api_key_file)
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
        # Whether the LAST fetch failed, so a failure is logged once per state
        # CHANGE rather than once per 5 s poll (see `gpus`). None = nothing
        # fetched yet, so the first failure of the process still logs.
        self._failing: bool | None = None

    def _auth_headers(self) -> dict[str, str]:
        """The bearer ``/api/gpu/detailed`` REQUIRES — dashboard-api's
        ``security.py`` has no unauthenticated path — from the env var if the
        install sets one, else from the key FILE dashboard-api mints on a
        stock install.

        Without the file arm, a stock install (``DASHBOARD_API_KEY`` unset)
        401s forever: dashboard-api generates a random key into
        ``/data/dashboard-api-key.txt`` and nothing else ever learns it. The
        dashboard's own nginx entrypoint reads exactly that file
        (extensions/services/dashboard/entrypoint.sh:5-20); this is the same
        fallback over the deck's ro ``/ods-data`` mount (compose.yaml).

        Read PER FETCH (so at most once per TTL window) rather than once at
        construction, because the two containers start together: on a fresh
        box the file does not exist yet when the deck builds its client, and
        a construction-time read would leave the deck 401ing until someone
        restarted it. A missing or unreadable file means no header at all —
        the same request this made before the fallback existed.
        """
        if self._key:
            return {"Authorization": f"Bearer {self._key}"}
        try:
            key = self._key_file.read_text().strip()
        except OSError:
            # Narrow I/O-boundary catch (repo CLAUDE.md): unmounted /ods-data,
            # a dashboard-api that has not written the file yet, or a
            # permission refusal all mean the same thing — no key here.
            return {}
        return {"Authorization": f"Bearer {key}"} if key else {}

    def gpus(self) -> list[dict] | None:
        if not self._url:
            return None
        now = self._clock()
        if self._fetched_at is not None and now < self._fetched_at + self._ttl_s:
            return self._cached
        try:
            resp = self._client.get("/api/gpu/detailed", headers=self._auth_headers())
            resp.raise_for_status()
            rows = resp.json()["gpus"]
            if not isinstance(rows, list):
                raise TypeError("gpus is not a list")
            result = _qualified(rows)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            # Narrow I/O-boundary catch (repo CLAUDE.md): transport failure,
            # a non-list "gpus", a missing "gpus" key, or a non-JSON body
            # all degrade to the same "no telemetry right now" answer — the
            # board already renders null gracefully, and this is a
            # best-effort observation, not a control path.
            #
            # Tolerated ⇒ LOGGED (same rule). Once per STATE CHANGE, not once
            # per poll: /api/state is polled every few seconds by every open
            # tab, so an unconditional line here would bury the log in
            # thousands of identical entries a day. A 401 in particular used
            # to be completely silent — a stock install whose
            # DASHBOARD_API_KEY is unset (see `_auth_headers`) showed an
            # empty stats block and said nothing, anywhere, ever.
            if not self._failing:
                _log.warning("local GPU telemetry unavailable: dashboard-api %s "
                             "%s: %s", self._url, type(exc).__name__, exc)
            self._failing = True
            result = None
        else:
            self._failing = False
        # A failed fetch is cached too (see module docstring) — the TTL
        # clock is what re-tries, not this branch.
        self._cached = result
        self._fetched_at = now
        return result
