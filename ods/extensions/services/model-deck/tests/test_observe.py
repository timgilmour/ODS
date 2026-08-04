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

    result = observe_spark(status)

    assert result["sparky/slot0"] == {
        "reachable": True, "loaded": True, "model": "heretic", "transitioning": False,
    }


def test_spark_profile_serving_under_a_different_name_is_not_drift():
    status = {"profile": "mm27b", "serving": {"model": "aeon"}, "reachable": True}

    result = observe_spark(status)

    assert result["sparky/slot0"]["model"] == "mm27b"


def test_spark_reachable_but_nothing_serving():
    status = {"profile": "heretic", "serving": None, "reachable": True}

    result = observe_spark(status)

    assert result["sparky/slot0"] == {
        "reachable": True, "loaded": False, "model": None, "transitioning": False,
    }


def test_spark_absent_client_yields_nothing():
    """No spark configured is not the same as spark being down — emit no
    key at all rather than a phantom unreachable resource."""
    assert observe_spark(None) == {}


def test_spark_unreachable():
    status = {"profile": None, "serving": None, "reachable": False}

    result = observe_spark(status)

    assert result["sparky/slot0"]["reachable"] is False


def test_merge_combines_maps():
    merged = merge_observations({"a": {"loaded": True}}, {"b": {"loaded": False}})

    assert set(merged) == {"a", "b"}


def test_engine_for_known_and_unknown_keys():
    from app.observe import engine_for

    assert engine_for("local/hipfire") == "hipfire"
    assert engine_for("sparky/slot0") == "spark"
    assert engine_for("nope/nothing") is None
