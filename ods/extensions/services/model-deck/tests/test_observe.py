"""Tests for app.observe — engine-specific snapshots -> one flat shape.

Each engine reports its own vocabulary (lemonade 'loaded'/'unloaded',
hipfire 'running'/'stopped', comfyui 'busy'/'idle', all of them 'unknown'
on an EngineError). This module is the ONLY place those differences live;
everything downstream sees {reachable, loaded, model}.
"""

from app.observe import merge_observations, observe_local, observe_spark


def _world(lemonade_state="loaded", hipfire_state="running", comfy_state="idle"):
    return {
        "gpus": [],
        "tenants": {
            "lemonade": {"state": lemonade_state, "model": "qwen", "footprint": 1, "idle_s": 0},
            "hipfire": {"state": hipfire_state, "model": "gpt-oss", "footprint": 1, "queue_depth": 0},
            "comfyui": {"state": comfy_state, "queue": 0, "idle_s": 0},
        },
        "externals": [],
        "default_route": None,
    }


def test_lemonade_loaded_maps_to_loaded():
    result = observe_local(_world())

    assert result["local/lemonade"] == {
        "reachable": True, "loaded": True, "model": "qwen", "transitioning": False,
    }


def test_lemonade_unloaded_maps_to_not_loaded():
    result = observe_local(_world(lemonade_state="unloaded"))

    assert result["local/lemonade"]["loaded"] is False
    assert result["local/lemonade"]["reachable"] is True


def test_observe_local_lemonade_loading_is_transitioning():
    """'loading' means a llama-server load is in flight — health reports
    nothing loaded while weights stream in. Reporting it as plain not-loaded
    would make the reconciler restart a model that is already coming up."""
    result = observe_local(_world(lemonade_state="loading"))

    assert result["local/lemonade"]["transitioning"] is True
    assert result["local/lemonade"]["loaded"] is False

    # ...and the derived status is the inert 'warming', per derive_status.
    from app.lifecycle import derive_status

    status = derive_status(
        {"state": "loaded", "model": "extra.m.gguf", "engine": "lemonade"},
        result["local/lemonade"],
    )
    assert status["status"] == "warming"


def test_hipfire_running_maps_to_loaded():
    result = observe_local(_world())

    assert result["local/hipfire"] == {
        "reachable": True, "loaded": True, "model": "gpt-oss", "transitioning": False,
    }


def test_hipfire_parked_maps_to_not_loaded_but_reachable():
    """A parked container is a reachable node with nothing loaded — the
    Deck can see it perfectly well. Only 'unknown' means we couldn't look."""
    result = observe_local(_world(hipfire_state="parked"))

    assert result["local/hipfire"] == {
        "reachable": True, "loaded": False, "model": None, "transitioning": False,
    }


def test_hipfire_loading_is_flagged_transitioning():
    """'loading' means the container is up but not yet healthy. Reporting it
    as plain not-loaded would make the reconciler restart a model that is
    already coming up — a restart storm out of a normal boot."""
    result = observe_local(_world(hipfire_state="loading"))

    assert result["local/hipfire"]["transitioning"] is True
    assert result["local/hipfire"]["loaded"] is False


def test_unknown_state_maps_to_unreachable():
    result = observe_local(_world(lemonade_state="unknown"))

    assert result["local/lemonade"]["reachable"] is False


def test_comfyui_busy_and_idle_both_count_as_loaded():
    """ComfyUI holds VRAM whether or not it is mid-job; 'idle' is not
    'unloaded' for it, which is exactly the kind of per-engine quirk this
    module exists to absorb."""
    assert observe_local(_world(comfy_state="busy"))["local/comfyui"]["loaded"] is True
    assert observe_local(_world(comfy_state="idle"))["local/comfyui"]["loaded"] is True


def test_comfyui_unknown_is_unreachable():
    assert observe_local(_world(comfy_state="unknown"))["local/comfyui"]["reachable"] is False


def test_spark_identity_is_the_profile_not_the_served_name():
    """The Deck's unit of control on spark is the PROFILE (that is what
    swap takes), so that is the identity intent is compared against. Using
    the served model name would report permanent drift the moment a profile
    served under a different --served-model-name."""
    status = {"profile": "heretic", "serving": {"model": "heretic"}, "reachable": True}

    result = observe_spark(status, "sparky")

    assert result["sparky/slot0"] == {
        "reachable": True, "loaded": True, "model": "heretic", "transitioning": False,
    }


def test_spark_profile_serving_under_a_different_name_is_not_drift():
    status = {"profile": "mm27b", "serving": {"model": "aeon"}, "reachable": True}

    result = observe_spark(status, "sparky")

    assert result["sparky/slot0"]["model"] == "mm27b"


def test_spark_reachable_but_nothing_serving():
    status = {"profile": "heretic", "serving": None, "reachable": True}

    result = observe_spark(status, "sparky")

    assert result["sparky/slot0"] == {
        "reachable": True, "loaded": False, "model": None, "transitioning": False,
    }


def test_spark_absent_client_yields_nothing():
    """No spark configured is not the same as spark being down — emit no
    key at all rather than a phantom unreachable resource."""
    assert observe_spark(None, "sparky") == {}


def test_spark_unreachable():
    status = {"profile": None, "serving": None, "reachable": False}

    result = observe_spark(status, "sparky")

    assert result["sparky/slot0"]["reachable"] is False


def test_merge_combines_maps():
    merged = merge_observations({"a": {"loaded": True}}, {"b": {"loaded": False}})

    assert set(merged) == {"a", "b"}


def test_engine_for_known_and_unknown_keys():
    from app.observe import engine_for

    assert engine_for("local/hipfire") == "hipfire"
    assert engine_for("boxa/slot0", frozenset({"boxa"})) == "spark"
    assert engine_for("nope/nothing") is None


# ===========================================================================
# SparkObserver — the one cached, backed-off translation of the node payload
# ===========================================================================


class _CountingSpark:
    """Counts status() calls and returns the NODE payload shape, so a test
    exercises the same translation production does."""

    def __init__(self, profile="heretic", serving_model="heretic",
                 endpoint_ok=True, raises=None):
        self.calls = 0
        self.raises = raises
        self._payload = {
            "profiles": [{"name": profile, "engine": "vllm"}],
            "swap_status": {"state": "error", "profile": profile, "ts": None},
            "serving": {"model": serving_model, "endpoint_ok": endpoint_ok},
        }

    def status(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self._payload


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _observer(spark, clock=None, **kw):
    from app.observe import SparkObserver

    return SparkObserver(lambda: spark, clock=clock or _Clock(), **kw)


def test_spark_observer_translates_the_node_payload():
    """SparkClient.status() returns {'profiles','swap_status','serving'} —
    NOT the observation shape. Identity is the PROFILE, not the served name."""
    spark = _CountingSpark(profile="heretic", serving_model="aeon")

    status = _observer(spark).status()

    assert status["profile"] == "heretic"
    assert status["reachable"] is True
    assert status["swap_in_progress"] is False
    assert observe_spark(status, "sparky")["sparky/slot0"]["model"] == "heretic"


def test_spark_observer_returns_none_without_a_spark():
    assert _observer(None).status() is None


def test_spark_observer_caches_within_the_ttl():
    """watch_interval defaults to 2 s; probing the node twice a tick is how
    the Deck went sluggish."""
    clock = _Clock()
    spark = _CountingSpark()
    obs = _observer(spark, clock=clock, ttl_s=10.0)

    obs.status()
    clock.advance(9.0)
    obs.status()

    assert spark.calls == 1


def test_spark_observer_reprobes_after_the_ttl():
    clock = _Clock()
    spark = _CountingSpark()
    obs = _observer(spark, clock=clock, ttl_s=10.0)

    obs.status()
    clock.advance(11.0)
    obs.status()

    assert spark.calls == 2


def test_spark_observer_backs_off_after_failures():
    """An absent sparky is its NORMAL state. Each failed probe costs two 5 s
    httpx timeouts, so a tick-rate probe stretches a 2 s cadence to ~12 s."""
    from app.engines import EngineError

    clock = _Clock()
    spark = _CountingSpark(raises=EngineError("connection refused"))
    obs = _observer(spark, clock=clock, ttl_s=10.0, backoff_base_s=15.0)

    obs.status()
    clock.advance(11.0)          # past the TTL, inside the backoff
    status = obs.status()

    assert spark.calls == 1
    assert status["reachable"] is False


def test_spark_observer_backoff_grows_and_is_capped():
    from app.engines import EngineError

    clock = _Clock()
    spark = _CountingSpark(raises=EngineError("down"))
    obs = _observer(spark, clock=clock, backoff_base_s=10.0, backoff_max_s=25.0)

    obs.status()                 # probe 1 -> wait 10
    clock.advance(10.0)
    obs.status()                 # probe 2 -> wait 20
    clock.advance(10.0)
    obs.status()                 # still inside the second backoff
    assert spark.calls == 2
    clock.advance(10.0)
    obs.status()                 # probe 3 -> wait 40, capped to 25
    assert spark.calls == 3
    clock.advance(25.0)
    obs.status()

    assert spark.calls == 4


def test_spark_observer_backoff_survives_days_of_failures():
    """~3.5 days of failed probes (>1024 at the 300 s cap) must keep
    returning the unreachable placeholder, not raise OverflowError from
    15.0 * 2**huge — 2**(failures-1) is a Python bignum and float
    multiplication against it overflows well before a real node stays down
    that long."""
    from app.engines import EngineError

    clock = _Clock()
    spark = _CountingSpark(raises=EngineError("connection refused"))
    obs = _observer(spark, clock=clock)

    for _ in range(1200):
        status = obs.status()
        assert status == {"profile": None, "serving": None,
                          "reachable": False, "swap_in_progress": False}
        clock.advance(301.0)         # past every possible backoff


def test_spark_observer_backoff_schedule_matches_production_defaults():
    """Pins the exponent cap (16) against PRODUCTION's real
    backoff_base_s=15/backoff_max_s=300 -- every other backoff test in this
    file uses small custom base/max values that never come near the cap, so
    a mutant like ``min(self._failures - 1, 1)`` would leave the whole
    suite green. The 5th failure's own wait is 15 * 2**4 = 240 s
    (uncapped); the 6th's would be 15 * 2**5 = 480 s, capped to 300 s."""
    from app.engines import EngineError

    clock = _Clock()
    spark = _CountingSpark(raises=EngineError("down"))
    obs = _observer(spark, clock=clock)   # production defaults: base 15, max 300

    for wait in (15.0, 30.0, 60.0, 120.0):   # failures 1 -> 4, uncapped 15*2**(n-1)
        obs.status()
        clock.advance(wait)
    obs.status()                          # failure #5
    assert spark.calls == 5

    clock.advance(239.0)
    obs.status()                          # not yet due -- still the 5th probe's cache
    assert spark.calls == 5
    clock.advance(1.0)                    # the 5th failure's 240 s wait, exactly elapsed
    obs.status()                          # failure #6
    assert spark.calls == 6

    clock.advance(299.0)
    obs.status()                          # not yet due -- still the 6th probe's cache
    assert spark.calls == 6
    clock.advance(1.0)                    # the capped 300 s wait, exactly elapsed
    obs.status()
    assert spark.calls == 7


def test_spark_observer_recovers_immediately_after_a_success():
    from app.engines import EngineError

    clock = _Clock()
    spark = _CountingSpark(raises=EngineError("down"))
    obs = _observer(spark, clock=clock, ttl_s=10.0, backoff_base_s=15.0)

    obs.status()
    spark.raises = None
    clock.advance(16.0)
    obs.status()                 # recovers: back to the short TTL
    clock.advance(11.0)
    obs.status()

    assert spark.calls == 3


def test_spark_observer_hands_the_error_to_its_caller_once():
    """The watcher owns the audit log, so the observer parks the error and
    the watcher takes it — probing from an HTTP thread must not double-log."""
    from app.engines import EngineError

    spark = _CountingSpark(raises=EngineError("connection refused"))
    obs = _observer(spark)

    obs.status()

    assert "connection refused" in obs.take_error()
    assert obs.take_error() is None


def test_spark_observer_invalidate_forces_a_fresh_probe():
    """A swap just changed the thing being cached."""
    clock = _Clock()
    spark = _CountingSpark()
    obs = _observer(spark, clock=clock, ttl_s=60.0)

    obs.status()
    obs.invalidate()
    obs.status()

    assert spark.calls == 2


# ===========================================================================
# N1 T3 — node-parametrized key vocabulary
# ===========================================================================

from app.observe import engine_for, observe_spark, slot_key


def test_slot_key_is_node_slash_slot0():
    assert slot_key("boxa") == "boxa/slot0"


def test_observe_spark_emits_the_given_nodes_key():
    status = {"profile": "laguna", "serving": {"model": "aeon"},
              "reachable": True, "swap_in_progress": False}
    obs = observe_spark(status, "boxa")
    assert set(obs) == {"boxa/slot0"}
    assert obs["boxa/slot0"]["loaded"] is True
    assert obs["boxa/slot0"]["model"] == "laguna"   # identity is the PROFILE


def test_observe_spark_none_emits_no_key_for_any_node():
    assert observe_spark(None, "boxa") == {}


def test_engine_for_swap_key_requires_declared_node():
    assert engine_for("boxa/slot0", frozenset({"boxa", "boxb"})) == "spark"
    assert engine_for("boxa/slot0", frozenset()) is None
    assert engine_for("boxa/slot0", frozenset({"boxb"})) is None


def test_engine_for_only_slot0_maps_to_spark():
    assert engine_for("boxa/slot1", frozenset({"boxa"})) is None
    assert engine_for("boxa", frozenset({"boxa"})) is None


def test_engine_for_local_keys_ignore_the_id_set():
    assert engine_for("local/hipfire") == "hipfire"
    assert engine_for("local/lemonade", frozenset({"local"})) == "lemonade"
