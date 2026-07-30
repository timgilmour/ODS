"""deck-drill live suite — fixtures and guarantees.

Runs INSIDE a throwaway container on ods-network (see ../deck-drill).
Everything here talks to the LIVE box; every mutating fixture restores
in its finalizer, and the session bookend fails the run if the box's
end state drifted from its start state.
"""

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, "/suite")  # extension root: makes `from app.gpu import ...` work

from clients import (Catalog, ComfyDirect, EventCursor, LemonadeDirect,  # noqa: E402
                     LitellmDirect, EXTRA)

DECK_URL = os.environ.get("DECK_URL", "http://model-deck:3015")
LEMONADE_URL = os.environ.get("LEMONADE_URL", "http://llama-server:8080")
COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188")
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm:4000")
HIPFIRE_URL = os.environ.get("HIPFIRE_URL", "http://ods-hipfire:11435")
CATALOG_URL = os.environ.get("CATALOG_URL", "http://dashboard-api:3001")
GGUF_STORE = Path("/gguf-store")

# Watcher tick is 2 s; TTL drills wait TTL + 3 ticks + margin.
TICK = 2.0


def wait_bound(ttl: float) -> float:
    return ttl + 3 * TICK + 4


@pytest.fixture(scope="session")
def deck() -> httpx.Client:
    with httpx.Client(base_url=DECK_URL, timeout=10.0) as client:
        yield client


@pytest.fixture(scope="session")
def lemonade_direct():
    return LemonadeDirect(LEMONADE_URL, os.environ.get("LEMONADE_API_KEY", ""))


@pytest.fixture(scope="session")
def comfy_direct():
    return ComfyDirect(COMFY_URL)


@pytest.fixture(scope="session")
def litellm_direct():
    return LitellmDirect(LITELLM_URL, os.environ.get("LITELLM_KEY", ""))


@pytest.fixture(scope="session")
def catalog():
    return Catalog(CATALOG_URL)


@pytest.fixture(scope="session")
def drill_model() -> str:
    """Bare filename of the smallest GGUF in the store (~1.2 G today)."""
    ggufs = sorted(GGUF_STORE.glob("*.gguf"), key=lambda p: p.stat().st_size)
    assert ggufs, "no GGUFs in /gguf-store"
    return ggufs[0].name


@pytest.fixture
def events(deck) -> EventCursor:
    return EventCursor(deck)


@pytest.fixture
def policy_guard(deck):
    """Yield the current policy; restore it byte-identical afterwards."""
    before = deck.get("/api/policy").json()
    yield before
    deck.put("/api/policy", json=before).raise_for_status()


@pytest.fixture
def lemonade_guard(deck, lemonade_direct):
    """Yield the pre-test loaded model (with extra. prefix, or None); restore it."""
    before = lemonade_direct.loaded()
    yield before
    after = lemonade_direct.loaded()
    if after == before:
        return
    if before is None:
        deck.post("/api/tenants/lemonade/unload", json={"model": None})
    else:
        deck.post("/api/tenants/lemonade/load",
                  json={"model": before.removeprefix(EXTRA)},
                  timeout=240.0).raise_for_status()


@pytest.fixture
def require_comfy_idle(comfy_direct):
    if comfy_direct.queue_len() != 0:
        pytest.skip("busy-skip: comfy queue is not empty")


# ---------------------------------------------------------------------------
# Run report: one md + one json per run in /reports (wrapper mounts the dir).
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

_RESULTS: list[dict] = []
_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    _RESULTS.append({
        "test": report.nodeid.split("::")[-1],
        "outcome": report.outcome,
        "reason": str(report.longrepr) if report.skipped else
                  (str(report.longrepr)[:500] if report.failed else ""),
        "duration_s": round(report.duration, 1),
    })


def pytest_sessionfinish(session, exitstatus):
    reports = Path("/reports")
    if not reports.is_dir():
        return
    (reports / f"{_STAMP}.json").write_text(json.dumps(
        {"exitstatus": int(exitstatus), "results": _RESULTS}, indent=2))
    lines = [f"# deck-drill {_STAMP}", ""]
    for r in _RESULTS:
        mark = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[r["outcome"]]
        extra = f" — {r['reason']}" if r["outcome"] != "passed" and r["reason"] else ""
        lines.append(f"- {mark} `{r['test']}` ({r['duration_s']}s){extra}")
    bookend = reports / "bookend.json"
    lines += ["", "## Left-as-found",
              "clean" if (not bookend.exists() or bookend.read_text() in ("{}", ""))
              else f"**DRIFT**: {bookend.read_text()}"]
    (reports / f"{_STAMP}.md").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="session", autouse=True)
def box_bookend(deck, lemonade_direct):
    """Left-as-found proof: fail the run if the box's end state drifted."""
    opening = {
        "loaded": lemonade_direct.loaded(),
        "hipfire": deck.get("/api/state").json()["world"]["tenants"]["hipfire"]["state"],
        "policy": deck.get("/api/policy").json(),
    }
    yield opening
    closing = {
        "loaded": lemonade_direct.loaded(),
        "hipfire": deck.get("/api/state").json()["world"]["tenants"]["hipfire"]["state"],
        "policy": deck.get("/api/policy").json(),
    }
    diff = {k: (opening[k], closing[k]) for k in opening if opening[k] != closing[k]}
    reports = Path("/reports")
    if reports.is_dir():
        (reports / "bookend.json").write_text(json.dumps(diff, indent=2))
    assert not diff, f"box NOT left as found: {diff}"
