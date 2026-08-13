"""Per-resource actuation/observation clients for DECLARED local engines,
the LocalClients counterpart to app.node_clients.NodeClients (swap nodes).

Mirrors NodeClients' lazy, self-healing pattern exactly (see that module's
docstring): a client is built on first use from the CURRENT declaration
entry's connection, and rebuilt (old one dropped, never closed — none of
today's engine client classes expose a close(), so there is nothing new to
leak here that _build_deck's one-shot construction didn't already leave
unclosed) whenever the entry's `(kind, connection)` pair changes, or
dropped entirely when the resource disappears from the declaration.
`client_for` answers None — never raises — for a resource that isn't
declared, exactly NodeClients' "not operable right now is a state, not an
error" posture.

Construction dispatches by kind (`app.engine_kinds.KNOWN_KINDS` names the
valid set; this module trusts NodeStore already validated `entry["kind"]`
via `engine_kinds.validate_engines` before it ever reaches here) to the
EXACT constructor calls app.main._build_deck used to make once, moved
here verbatim rather than reinvented. This is a small, disclosed residue
of engine-kind-name literals outside app/engine_kinds.py (see that
module's docstring and the plan's Global Constraints on residues) — the
per-kind constructor SHAPES (which class, which args, built from which
connection fields vs. which shared Settings fields) are irreducibly
different, so there is no single dispatch table engine_kinds.py could hand
back without effectively re-exporting these same client classes.

hipfire-kind construction builds its own DockerCtl/LiteLLMClient from
Settings rather than reusing app.main._build_deck's shared `dockerctl`/
`litellm` instances (those still exist in the deck dict today, feeding
actuation — control.py's routes and the watcher's `self._hipfire` — which
Task 3 deliberately leaves untouched, COEXISTENCE: observation only). Both
classes are stateless besides their own httpx.Client, so a second instance
behaves identically for `status()`/`stats()` reads; the one known
transitional gap is HipfireClient's own conversation-activity tracker
(fed by every `stats()` call, read by `ensure_not_busy`'s recency check) —
this module's instance and the deck's shared `hipfire` instance now poll
independently, so the recency half of the busy guard is only as fresh as
whichever instance last polled. Acceptable for this increment (actuation
is untouched here; Task 6 migrates control.py/the watcher onto this same
LocalClients, closing the gap) — flagged for visibility, not silently
absorbed.
"""

from __future__ import annotations

import threading

from app.engines.comfyui import ComfyClient
from app.engines.docker_ctl import DockerCtl
from app.engines.hipfire import HipfireClient
from app.engines.lemonade import LemonadeClient
from app.engines.litellm import LiteLLMClient

# hipfire runs as a sibling container on the compose network; its health
# endpoint is <container>:11435/health (config/ports.json + manifest.yaml).
# Mirrors app.main's own _HIPFIRE_PORT constant (that module's is not
# imported here to avoid a local_clients -> main import cycle).
_HIPFIRE_PORT = 11435


def _build_key(entry: dict) -> tuple:
    return (entry["kind"], frozenset(entry["connection"].items()))


def _build_client(entry: dict, settings):
    conn = entry["connection"]
    kind = entry["kind"]
    if kind == "lemonade":
        return LemonadeClient(conn["url"], settings.lemonade_key,
                              metrics_url=conn["metrics_url"])
    if kind == "comfyui":
        return ComfyClient(conn["url"])
    if kind == "hipfire":
        container = conn["container"]
        dockerctl = DockerCtl(settings.dockerctl_url, settings.park_allowlist)
        litellm = LiteLLMClient(settings.litellm_url, settings.litellm_key)
        return HipfireClient(
            health_url=f"http://{container}:{_HIPFIRE_PORT}/health",
            dockerctl=dockerctl,
            container=container,
            litellm=litellm,
            stats_url=f"http://{container}:{_HIPFIRE_PORT}/stats",
            activity_window_s=settings.hipfire_activity_window_s,
        )
    # Unreachable in production: NodeStore validates `kind` against
    # engine_kinds.KNOWN_KINDS (app.engine_kinds.validate_engines) before an
    # entry can ever land in the declaration this class reads from.
    raise ValueError(f"unknown engine kind {kind!r}")


class LocalClients:
    """Lazy, self-healing map of resource -> declared local engine client
    (design §3 / Task 3 brief; mirrors app.node_clients.NodeClients)."""

    def __init__(self, node_store, settings):
        self._store = node_store
        self._settings = settings
        self._lock = threading.Lock()
        # resource -> (build_key, client) for clients actually built.
        self._built: dict[str, tuple[tuple, object]] = {}

    def client_for(self, resource: str):
        """The current client for `resource`, or None when it is not
        declared right now. Rebuilds when the declaration's (kind,
        connection) pair changed since the last build."""
        with self._lock:
            local = self._store.get("local")
            entry = None
            if local is not None:
                for e in local.get("engines", []):
                    if e["resource"] == resource:
                        entry = e
                        break
            if entry is None:
                self._retire(resource)
                return None
            key = _build_key(entry)
            built = self._built.get(resource)
            if built is not None and built[0] == key:
                return built[1]
            self._retire(resource)
            client = _build_client(entry, self._settings)
            self._built[resource] = (key, client)
            return client

    def _retire(self, resource: str) -> None:
        self._built.pop(resource, None)

    def retire_absent(self, keep_resources) -> None:
        """Drop built clients whose resource is not in `keep_resources` —
        called after a declaration edit removes/renames a resource, so a
        stale client is never handed out again. Mirrors NodeClients'
        method of the same name (app/node_clients.py)."""
        with self._lock:
            for resource in list(self._built):
                if resource not in keep_resources:
                    self._retire(resource)
