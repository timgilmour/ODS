"""Tests for app.validate_settings — three warning classes, zero blocks.

Nothing here prevents a save. You must always be able to reach a flag a new
engine version added before we re-harvested; the cost of blocking is a
person who cannot do their job, and the cost of warning is a yellow chip.
"""

from app.validate_settings import validate_settings

CATALOG = {
    "options": {
        "max-model-len": {"type": "int", "choices": None, "nargs": None,
                          "widget": "number", "aliases": [], "default": "None", "help": ""},
        "quantization": {"type": "str", "choices": ["modelopt", "compressed-tensors"],
                         "nargs": None, "widget": "select", "aliases": ["-q"],
                         "default": "None", "help": ""},
    },
    "engine_version": "0.26.0",
}


def _resolved(**kwargs):
    # AMENDED 2026-08-07 (Task 3 review): resolved values are GUARANTEED
    # normalized (str, True, or list[str] — never int/float, never a
    # singleton list) because store layers normalize on write and Task 7's
    # _resolve() passes derived layers through argline.normalize_args_map.
    # Validators may rely on that invariant; validate_settings should still
    # degrade gracefully (warn, never crash) if handed a raw shape anyway.
    return {k: {"value": v, "origin": "declared", "layer": "engine"} for k, v in kwargs.items()}


def _facts(**kwargs):
    return {k: {"value": v, "origin": "derived", "source": "config.json", "derived_ts": "t"}
            for k, v in kwargs.items()}


def test_known_flag_with_valid_value_produces_no_warning():
    assert validate_settings(_resolved(**{"max-model-len": "131072"}), CATALOG, {}) == []


def test_unknown_flag_warns_but_is_never_an_error():
    warnings = validate_settings(_resolved(**{"brand-new-flag": "1"}), CATALOG, {})

    assert len(warnings) == 1
    assert warnings[0]["class"] == "unknown"
    assert warnings[0]["severity"] == "warn"


def test_no_catalog_means_everything_is_unvalidated_and_still_fine():
    """An engine that has never been up has no catalog. Every flag reads
    'unvalidated' and stays editable — this is a supported state."""
    warnings = validate_settings(_resolved(**{"max-model-len": "1"}), None, {})

    assert [w["class"] for w in warnings] == ["unknown"]


def test_wrong_type_warns():
    warnings = validate_settings(_resolved(**{"max-model-len": "not-a-number"}), CATALOG, {})

    assert warnings[0]["class"] == "type"


def test_value_outside_choices_warns():
    warnings = validate_settings(_resolved(quantization="banana"), CATALOG, {})

    assert warnings[0]["class"] == "type"
    assert "modelopt" in warnings[0]["message"]


def test_quantization_conflicting_with_the_checkpoint_is_a_crash_conflict():
    """THE regression: heretic and ornith crash-looped for weeks on exactly
    this, and it was invisible until someone read the logs."""
    warnings = validate_settings(
        _resolved(quantization="modelopt"),
        CATALOG,
        _facts(quant_method="compressed-tensors"),
    )

    conflicts = [w for w in warnings if w["class"] == "conflict"]
    assert conflicts[0]["severity"] == "crash"
    assert "compressed-tensors" in conflicts[0]["message"]


def test_quantization_agreeing_with_the_checkpoint_is_silent():
    """mm27b genuinely IS modelopt. A false positive here would teach
    everyone to ignore the warnings."""
    warnings = validate_settings(
        _resolved(quantization="modelopt"), CATALOG, _facts(quant_method="modelopt"))

    assert [w for w in warnings if w["class"] == "conflict"] == []


def test_no_facts_means_no_conflict_warnings():
    """Remote models have no readable checkpoint. Cannot check is not
    mismatch."""
    warnings = validate_settings(_resolved(quantization="modelopt"), CATALOG, {})

    assert [w for w in warnings if w["class"] == "conflict"] == []


def test_positional_key_is_not_validated():
    from app.argline import POSITIONAL_KEY

    resolved = {POSITIONAL_KEY: {"value": ["serve", "/model"],
                                 "origin": "declared", "layer": "engine"}}

    assert validate_settings(resolved, CATALOG, {}) == []


def test_every_warning_names_its_key():
    warnings = validate_settings(_resolved(**{"nope": "1"}), CATALOG, {})

    assert warnings[0]["key"] == "nope"


def test_warnings_never_raise_regardless_of_input_shape():
    validate_settings({}, None, {})
    validate_settings(_resolved(a=True), CATALOG, {})
