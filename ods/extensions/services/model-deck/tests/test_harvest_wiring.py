"""Remote harvest wiring: the spark catalog cache satisfies C1's
engine_exec protocol, so Watcher._harvest_catalogs runs UNCHANGED — one
parser, one version-skip, one storage seam (arbiter.py:904-950).

The Watcher end-to-end tests (real sentinel + real parse_probe_output) live
in tests/test_arbiter.py's CATALOG HARVEST section, alongside the rest of
_harvest_catalogs' coverage and its `_watcher`/`_make_watcher` construction
helpers -- these two are the unit-level pieces underneath that seam:
SparkCatalogExec's own contract, and the router that dispatches to it.
"""

import pytest

from app.engines import EngineError
from app.engines.docker_ctl import EngineExecRouter
from app.engines.spark import SparkCatalogExec


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
