"""The node-observation pass, on its OWN thread.

READS THE REGISTRY, WRITES A SNAPSHOT. It holds no intent store, no docker
client, and a client with no verbs (NodeAgentClient) — it structurally
cannot become a second actuator, the same argument app/update_check.py
makes for its thread. NEVER called from arbiter.Watcher.tick(): that tick
is one synchronous thread running the reconciler, and N nodes x a 5 s
transport timeout on a down box would stall the machinery that keeps
models loaded.

The registry is RE-READ every pass (dashboard-api remote_nodes.py's 5 s
re-read precedent) so add/remove/credential changes apply live, no restart.

Status vocabulary — produced HERE and nowhere else; the UI compares these
exact strings (ui/src/api.ts NodeAgentStatus mirrors them):
  online       — the gpu probe answered (remote_nodes.py:12-19: that probe
                 ALONE governs; a failed serving probe only degrades
                 `serving` to None)
  offline      — transport failure (NodeAgentUnreachable)
  error        — answered badly (non-2xx / bad body: plain EngineError)
  unconfigured — no stored credential; never probed. Distinct from offline:
                 "not set up" and "not answering" must not collapse.

Sparky is observed twice on purpose: the lifecycle path (SparkObserver ->
observe_spark -> derive_status) keeps answering "is the slot serving" for
the reconciler and the board's spark card; this pass answers "is the box
answering" for the Nodes screen. Two consumers, two questions — v1 does
not route the reconciler through any new observation path.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from app import events
from app.engines import EngineError
from app.engines.node_agent import NodeAgentClient, NodeAgentUnreachable

_UNCONFIGURED_NOTE = "no credential stored — edit the node to add one"


class NodeObserver:
    def __init__(self, node_store, events_path: Path, interval: float = 10.0,
                 client_factory=None):
        self._store = node_store
        self._events_path = events_path
        self._interval = interval
        self._client_factory = client_factory or (
            lambda address, key: NodeAgentClient(address, key))
        self._snap: dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error_type: str | None = None

    # -- lifecycle (UpdateChecker idiom) ------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run,
                                        name="model-deck-node-observer",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 — supervisor loop, never silent
                # Class-name only, deduped on change: the arbiter.py:438
                # precedent — a permanently-broken store must not log a new
                # event every 10 s.
                name = type(exc).__name__
                if name != self._last_error_type:
                    self._last_error_type = name
                    events.log_event(self._events_path, "node-observe-error",
                                     {"note": name})
            if self._stop.wait(self._interval):
                break

    # -- the pass -----------------------------------------------------------

    def tick(self) -> None:
        previous = self._snap
        snap: dict[str, dict] = {}
        for entry in self._store.list():
            # Shape is guaranteed by NodeStore._load()'s boundary gate
            # (app/node_store.py): every entry here is a dict with a string
            # id/label and a known agent_kind. Only SEMANTICS remain to
            # check here, not shape.
            if entry["agent_kind"] != "node-agent":
                continue
            node_id = entry["id"]
            if not entry.get("address"):
                # address is legitimately optional shape-wise (local has
                # none); a node-agent entry missing one just can't be
                # probed — skip it, same as before the boundary gate.
                continue
            if not self._store.credential_set(node_id):
                snap[node_id] = {"status": "unconfigured", "last_seen": None,
                                 "gpus": None, "serving": None,
                                 "error": _UNCONFIGURED_NOTE}
                continue
            snap[node_id] = self._probe(entry,
                                        previous.get(node_id, {}).get("last_seen"))
        # One atomic reference swap; readers never see a half-built pass.
        self._snap = snap

    def _probe(self, entry: dict, previous_last_seen: str | None) -> dict:
        client = self._client_factory(entry["address"],
                                      self._store.credential_for(entry["id"]))
        try:
            try:
                gpu = client.gpu()
            except NodeAgentUnreachable as exc:
                return {"status": "offline", "last_seen": previous_last_seen,
                        "gpus": None, "serving": None, "error": str(exc)}
            except EngineError as exc:
                return {"status": "error", "last_seen": previous_last_seen,
                        "gpus": None, "serving": None, "error": str(exc)}
            try:
                serving = client.serving()
            except EngineError:
                serving = None    # auxiliary probe: degrades, never governs
            return {"status": "online",
                    "last_seen": datetime.now(UTC).isoformat(),
                    "gpus": gpu.get("gpus"), "serving": serving,
                    "error": gpu.get("error")}
        finally:
            client.close()

    def snapshot(self) -> dict[str, dict]:
        return self._snap
