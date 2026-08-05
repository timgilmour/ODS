"""Ground-truth clients: talk to the ENGINES and the kernel, not the deck.

Every assertion about "did the capability really happen" goes through here,
so a lying /api/state cannot self-certify.
"""

import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.gpu import read_gpus  # the deck's own reader → identical GPU indexing

DRM_ROOT = Path("/sysfs/class/drm")
KFD_ROOT = Path("/sysfs/class/kfd/kfd/proc")
EXTRA = "extra."


def read_vram(index: int) -> tuple[int, int]:
    gpus = read_gpus(DRM_ROOT, KFD_ROOT)
    gpu = next(g for g in gpus if g["index"] == index)
    return gpu["vram_total"], gpu["vram_used"]


class LemonadeDirect:
    def __init__(self, base_url: str, key: str):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._c = httpx.Client(base_url=base_url, headers=headers, timeout=10.0)

    def loaded(self) -> str | None:
        return self._c.get("/api/v1/health").json().get("model_loaded")

    def unload(self, model: str) -> None:
        r = self._c.post("/api/v1/unload", json={"model_name": model})
        r.raise_for_status()


class ComfyDirect:
    def __init__(self, base_url: str):
        self._c = httpx.Client(base_url=base_url, timeout=30.0)

    def queue_len(self) -> int:
        q = self._c.get("/queue").json()
        return len(q["queue_running"]) + len(q["queue_pending"])

    def submit(self, workflow: dict) -> str:
        r = self._c.post("/prompt", json={"prompt": workflow})
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait_idle(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.queue_len() == 0:
                return
            time.sleep(2)
        raise TimeoutError(f"comfy queue still busy after {timeout}s")


class LitellmDirect:
    def __init__(self, base_url: str, key: str):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._c = httpx.Client(base_url=base_url, headers=headers, timeout=120.0)

    def route_table(self) -> dict[str, str]:
        data = self._c.get("/model/info").json()["data"]
        return {e["model_name"]: e["litellm_params"]["model"] for e in data}

    def default_api_base(self) -> str:
        """`default`'s upstream api_base, or "" when there is no default route.

        The field the deck's own default-route guards read (both
        HipfireClient.park() and SparkClient.swap() match on api_base, not
        on the model name), so any drill skip condition derived from it
        cannot disagree with the guard that would refuse the call.
        """
        for entry in self._c.get("/model/info").json()["data"]:
            if entry["model_name"] == "default":
                return entry["litellm_params"].get("api_base", "") or ""
        return ""

    def default_targets_hipfire(self) -> bool:
        """The same predicate HipfireClient.park() guards on, read from the
        same field (api_base, not the model name) — so a drill's skip
        condition cannot disagree with the guard that would refuse it."""
        return "hipfire" in self.default_api_base()

    def completion(self, route: str, max_tokens: int = 64) -> dict:
        """One real served request; returns the response message dict.

        Ground truth is that the engine SERVED (2xx with a choice) — content may
        legitimately be empty when a thinking model (hipfire's xhigh) spends the
        whole token budget on reasoning, so callers assert on the returned dict,
        never on content text.
        """
        r = self._c.post("/v1/chat/completions", json={
            "model": route, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "Reply with the word: ok"}],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


class SparkServingDirect:
    """Ground truth for the Spark's single serving port (sparky:8000).

    Whichever profile is loaded owns that port exclusively — every spark
    profile runs ``network_mode: host`` — so this client doubles as the
    port-handover check: an answer identifying the INCOMING model is proof
    the outgoing container released the socket. That matters because the
    drill has no docker reach onto sparky at all (the node-agent
    deliberately holds no docker socket — the host-helper split, see
    app/engines/spark.py), so conftest's ``docker`` fixture cannot be
    pointed at it the way it can at local containers.
    """

    # Deliberately a literal here rather than an import of
    # app.engines.spark._BUSY_METRICS: the drill's job is to observe what
    # ds4 really emits. If the deck's copy of the gauge name ever drifts
    # from the server's, an independent literal is what makes the drill
    # notice instead of agreeing with the bug.
    DS4_INFLIGHT = "ds4_requests_inflight"

    def __init__(self, base_url: str):
        self._c = httpx.Client(base_url=base_url, timeout=30.0)
        # What the deck's litellm default-route guard matches on.
        self.host = urlparse(base_url).netloc

    def served_model(self) -> str | None:
        """The id on :8000, or None while nothing is listening there."""
        try:
            r = self._c.get("/v1/models", timeout=10.0)
        except httpx.TransportError:
            return None
        if not r.is_success:
            return None
        data = r.json().get("data") or []
        return data[0]["id"] if data else None

    def inflight(self) -> int | None:
        """ds4's in-flight count, or None when that gauge is absent.

        None is the load-bearing case, not an error: ds4 emits the gauge
        from a zero-initialized static registry on every scrape, so a 200
        without it means :8000 is being served by something that is not
        ds4 (or by nothing).
        """
        text = self._c.get("/metrics", timeout=10.0).text
        values = [float(line.rsplit(None, 1)[-1]) for line in text.splitlines()
                  if line.startswith(self.DS4_INFLIGHT)]
        return int(sum(values)) if values else None

    def completion(self, model: str, prompt: str, max_tokens: int,
                   timeout: float = 300.0) -> dict:
        """One real, blocking generation straight at the engine.

        Returns the whole response body so a caller can prove the request
        really produced tokens — "the swap was refused" only means
        something if there was genuinely something to refuse over.
        """
        r = self._c.post("/v1/chat/completions", timeout=timeout, json={
            "model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        return r.json()


def hipfire_health_status(url: str) -> str:
    try:
        httpx.get(f"{url}/health", timeout=5.0)
        return "up"
    except httpx.TransportError:
        return "down"


class Catalog:
    def __init__(self, base_url: str):
        self._c = httpx.Client(base_url=base_url, timeout=10.0)

    def id_for_gguf(self, filename: str) -> str | None:
        models = self._c.get("/api/models").json()["models"]
        return next((m["id"] for m in models if m.get("gguf") == filename), None)


class EventCursor:
    """Diff-based view of /api/events since construction."""

    def __init__(self, deck: httpx.Client, n: int = 1000):
        self._deck = deck
        self._n = n
        self._baseline = self._fetch()

    def _fetch(self) -> list[dict]:
        return self._deck.get("/api/events", params={"n": self._n}).json()["events"]

    def new_events(self) -> list[dict]:
        current = self._fetch()
        if not self._baseline:
            return current
        anchor = self._baseline[-1]
        for i in range(len(current) - 1, -1, -1):
            if current[i] == anchor:
                return current[i + 1:]
        return current  # anchor rotated out of the tail window

    def expect(self, kind: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            hit = next((e for e in self.new_events() if e["kind"] == kind), None)
            if hit:
                return hit
            time.sleep(1)
        raise AssertionError(f"no {kind!r} event within {timeout}s; saw "
                             f"{[e['kind'] for e in self.new_events()]}")

    def expect_absent(self, kind: str, window: float) -> None:
        time.sleep(window)
        hits = [e for e in self.new_events() if e["kind"] == kind]
        assert not hits, f"unexpected {kind!r} events: {hits}"

    def min_spacing(self, kind: str) -> float:
        """Smallest gap in seconds between consecutive events of `kind` (inf if <2)."""
        from datetime import datetime
        ts = [datetime.fromisoformat(e["ts"]) for e in self.new_events() if e["kind"] == kind]
        gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
        return min(gaps) if gaps else float("inf")
