"""Remote harvest wiring: the spark catalog cache satisfies C1's
engine_exec protocol, so Watcher._harvest_catalogs runs UNCHANGED — one
parser, one version-skip, one storage seam, now hoisted into module-level
``app.arbiter.harvest_catalog_pair`` (Task 3, C2/Phase 3) so the manual
force-harvest route (``POST /api/settings/harvest/{node}/{engine}``) can call
the exact same logic Watcher._harvest_catalogs loops over.

The Watcher end-to-end tests (real sentinel + real parse_probe_output) live
in tests/test_arbiter.py's CATALOG HARVEST section, alongside the rest of
_harvest_catalogs' coverage and its `_watcher`/`_make_watcher` construction
helpers -- these two are the unit-level pieces underneath that seam:
SparkCatalogExec's own contract, and the router that dispatches to it.

``test_harvest_pair_*`` below cover ``harvest_catalog_pair`` itself, reusing
the same fake-exec/PROBE_OUTPUT shape tests/test_arbiter.py's CATALOG HARVEST
section already established (a callable with a mutable ``.version``
attribute, PROBE_OUTPUT from tests/test_harvest.py) rather than inventing a
new fixture vocabulary.
"""

import pytest

from app.arbiter import harvest_catalog_pair
from app.characteristics import CharacteristicsStore
from app.engines import EngineError
from app.engines.docker_ctl import EngineExecRouter
from app.engines.spark import SparkCatalogExec
from tests.test_harvest import PROBE_OUTPUT


class _FakeExec:
    """Same shape as test_arbiter.py's ``_recording_exec``: a callable with
    a mutable ``.version`` (the cheap peek harvest_catalog_pair reads without
    invoking) that records every call and returns ``(self.version,
    PROBE_OUTPUT)``."""

    def __init__(self, version):
        self.version = version
        self.calls: list[tuple] = []

    def __call__(self, node, engine, interpreter, source):
        self.calls.append((node, engine, interpreter, source))
        return self.version, PROBE_OUTPUT


def test_spark_catalog_exec_returns_version_and_output():
    class FakeClient:
        def get_catalog(self):
            return {"catalog": {"image_id": "sha256:abc",
                                "harvested_ts": "2026-08-07T02:00:00Z",
                                "engine": "vllm", "probe_output": "SENTINEL..."}}

    version, output = SparkCatalogExec(FakeClient())("sparky", "vllm", "python3", "src")

    assert (version, output) == ("sha256:abc", "SENTINEL...")


def test_spark_catalog_exec_raises_engine_error_when_none_yet():
    class FakeClient:
        def get_catalog(self):
            return {"catalog": None}

    with pytest.raises(EngineError):
        SparkCatalogExec(FakeClient())("sparky", "vllm", "python3", "src")


def test_spark_catalog_exec_raises_engine_error_for_partial_catalog():
    """node-agent serves older-schema catalogs by design
    (node-agent/settings_store.py:88-110); a partial one must surface as the
    EngineError the harvest contract catches (arbiter.py:424-426), not
    KeyError."""
    class FakeClient:
        def get_catalog(self):
            return {"catalog": {"harvested_ts": "2026-08-10"}}

    with pytest.raises(EngineError):
        SparkCatalogExec(FakeClient())("sparky", "vllm", "python3", "src")


def test_router_dispatches_on_the_pair_and_rejects_unknown():
    calls = []
    router = EngineExecRouter({("sparky", "vllm"):
                               lambda n, e, i, s: calls.append((n, e)) or ("v", "o")})

    assert router("sparky", "vllm", "python3", "src") == ("v", "o")
    assert calls == [("sparky", "vllm")]
    with pytest.raises(EngineError):
        router("local", "vllm", "python3", "src")


def test_router_pairs_property_lists_the_routes():
    router = EngineExecRouter({
        ("sparky", "vllm"): lambda n, e, i, s: ("v", "o"),
        ("local", "hipfire"): lambda n, e, i, s: ("v2", "o2"),
    })

    assert sorted(router.pairs) == [("local", "hipfire"), ("sparky", "vllm")]


# ===========================================================================
# harvest_catalog_pair — the hoisted body of Watcher._harvest_catalogs' loop
# (Task 3, C2/Phase 3). Watcher's own coverage (test_arbiter.py's CATALOG
# HARVEST section) proves the loop still behaves byte-identically after the
# hoist; these two cover the `force=True` behavior only the manual endpoint
# exercises — that force skips BOTH version gates (the cheap peek AND the
# post-call compare), never just one.
# ===========================================================================


def test_harvest_pair_force_skips_version_gate(tmp_path):
    """A cached catalog whose engine_version equals the exec's peeked
    version is exactly what makes Watcher's own (force=False) path
    early-return "current" without ever calling engine_exec. force=True must
    skip that gate and harvest anyway — a human pressing Refresh on a
    catalog that only LOOKS current (peek matches, but the underlying probe
    output actually changed) is the whole point of the button."""
    store = CharacteristicsStore(tmp_path / "c.json")
    store.put_fields("engine/sparky/vllm", {"option_catalog": {
        "value": {"engine_version": "0.26.0", "options": {}},
        "source": "argparse introspection", "derived_ts": "t0",
    }})
    fake_exec = _FakeExec(version="0.26.0")

    result = harvest_catalog_pair(
        fake_exec, store, lambda kind, detail: None,
        "sparky", "vllm", now="2026-08-07T10:00:00+00:00", force=True)

    assert result["outcome"] == "harvested"
    assert result["options"] > 0
    assert len(fake_exec.calls) == 1  # force still pays for exactly one exec


def test_harvest_pair_without_force_reports_current(tmp_path):
    """Same cached-version-equals-peeked-version setup, force omitted
    (defaults False): the watcher's own early-return path, proving
    harvest_catalog_pair reproduces it exactly — no exec call at all."""
    store = CharacteristicsStore(tmp_path / "c.json")
    store.put_fields("engine/sparky/vllm", {"option_catalog": {
        "value": {"engine_version": "0.26.0", "options": {}},
        "source": "argparse introspection", "derived_ts": "t0",
    }})
    fake_exec = _FakeExec(version="0.26.0")

    result = harvest_catalog_pair(
        fake_exec, store, lambda kind, detail: None,
        "sparky", "vllm", now="2026-08-07T10:00:00+00:00")

    assert result["outcome"] == "current"
    assert fake_exec.calls == []
