"""Tests for app.ladder — five layers, per key, most specific wins.

Per-KEY merge is the property that matters. A whole-blob override is how
config systems rot: you set one flag at the most specific scope and silently
lose four you set higher up.
"""

from app.ladder import LAYERS, resolve_settings


def _resolve(**kwargs):
    base = {"engine_defaults": {}, "checkpoint_recommendations": {},
            "engine": {}, "model": {}, "engine_model": {}}
    base.update(kwargs)
    return resolve_settings(**base)


def test_layer_order_is_lowest_to_highest():
    assert LAYERS == ("engine_defaults", "checkpoint_recommendations",
                      "engine", "model", "engine_model")


def test_single_layer_value_passes_through():
    resolved = _resolve(engine={"max-model-len": "262144"})

    assert resolved["max-model-len"]["value"] == "262144"
    assert resolved["max-model-len"]["layer"] == "engine"


def test_more_specific_layer_wins():
    resolved = _resolve(engine={"max-model-len": "262144"},
                        engine_model={"max-model-len": "131072"})

    assert resolved["max-model-len"]["value"] == "131072"
    assert resolved["max-model-len"]["layer"] == "engine_model"


def test_merge_is_per_key_not_per_blob():
    """THE property. Overriding one key at the top layer must leave the
    other keys from lower layers intact."""
    resolved = _resolve(
        engine={"a": "1", "b": "2"},
        model={"c": "3"},
        engine_model={"b": "9"},
    )

    assert {k: v["value"] for k, v in resolved.items()} == {"a": "1", "b": "9", "c": "3"}


def test_derived_layers_are_marked_derived():
    resolved = _resolve(engine_defaults={"a": "1"}, checkpoint_recommendations={"b": "2"})

    assert resolved["a"]["origin"] == "derived"
    assert resolved["b"]["origin"] == "derived"


def test_user_layers_are_marked_declared():
    resolved = _resolve(engine={"a": "1"}, model={"b": "2"}, engine_model={"c": "3"})

    assert {resolved[k]["origin"] for k in "abc"} == {"declared"}


def test_checkpoint_recommendation_beats_engine_default():
    resolved = _resolve(engine_defaults={"top_k": "0"},
                        checkpoint_recommendations={"top_k": "20"})

    assert resolved["top_k"]["value"] == "20"


def test_user_engine_layer_beats_both_derived_layers():
    resolved = _resolve(engine_defaults={"top_k": "0"},
                        checkpoint_recommendations={"top_k": "20"},
                        engine={"top_k": "40"})

    assert resolved["top_k"]["value"] == "40"


def test_model_layer_beats_engine_layer():
    resolved = _resolve(engine={"a": "engine"}, model={"a": "model"})

    assert resolved["a"]["value"] == "model"


def test_all_empty_resolves_empty():
    assert _resolve() == {}


def test_a_bare_flag_is_a_value_like_any_other():
    resolved = _resolve(engine={"enable-prefix-caching": True})

    assert resolved["enable-prefix-caching"]["value"] is True


def test_explicit_none_at_a_higher_layer_unsets_a_lower_one():
    """Dropping an inherited flag needs a representation. None means 'do not
    pass this at all' — heretic's fix was the ABSENCE of --quantization, and
    the ladder has to be able to express that."""
    resolved = _resolve(engine={"quantization": "modelopt"},
                        engine_model={"quantization": None})

    assert "quantization" not in resolved


# --- review 2026-08-07 additions: minors, hand-verified correct beforehand ---


def test_same_key_present_in_all_five_layers_only_the_most_specific_wins():
    resolved = _resolve(
        engine_defaults={"a": "ed"},
        checkpoint_recommendations={"a": "cr"},
        engine={"a": "e"},
        model={"a": "m"},
        engine_model={"a": "em"},
    )

    assert resolved["a"]["value"] == "em"
    assert resolved["a"]["layer"] == "engine_model"


def test_key_only_in_lowest_layer_keeps_derived_origin_and_layer():
    resolved = _resolve(engine_defaults={"a": "1"})

    assert resolved["a"] == {"value": "1", "origin": "derived", "layer": "engine_defaults"}


def test_true_and_scalar_override_each_other_regardless_of_direction():
    bare_then_scalar = _resolve(engine={"a": True}, model={"a": "5"})
    scalar_then_bare = _resolve(engine={"a": "5"}, model={"a": True})

    assert bare_then_scalar["a"]["value"] == "5"
    assert scalar_then_bare["a"]["value"] is True


def test_equal_value_at_a_higher_layer_still_re_attributes_provenance():
    """Value-blind by design (confirmed correct in review): the ladder
    never compares values across layers, so even an unchanged value
    re-attributes to whichever layer most specifically set it — correct
    provenance, not a no-op skip."""
    resolved = _resolve(engine={"a": "5"}, engine_model={"a": "5"})

    assert resolved["a"]["value"] == "5"
    assert resolved["a"]["layer"] == "engine_model"
