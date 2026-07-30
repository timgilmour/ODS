"""Ground-truth clients: talk to the ENGINES and the kernel, not the deck.

Every assertion about "did the capability really happen" goes through here,
so a lying /api/state cannot self-certify.
"""

import time
from pathlib import Path

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

    def completion(self, route: str, max_tokens: int = 8) -> str:
        r = self._c.post("/v1/chat/completions", json={
            "model": route, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "Reply with the word: ok"}],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


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
