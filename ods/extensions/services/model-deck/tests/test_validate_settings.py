"""Tests for app.validate_settings — three warning classes, zero blocks.

Nothing here prevents a save. You must always be able to reach a flag a new
engine version added before we re-harvested; the cost of blocking is a
person who cannot do their job, and the cost of warning is a yellow chip.
"""

from app.validate_settings import validate_settings

CATALOG = {
    "options": {
        "max-model-len": {"type": "int", "choices": None, "nargs": None,
                          "widget": "number", "aliases": [], "default": "None", "help": "",
                          "repeatable": False},
        "quantization": {"type": "str", "choices": ["modelopt", "compressed-tensors"],
                         "nargs": None, "widget": "select", "aliases": ["-q"],
                         "default": "None", "help": "", "repeatable": False},
        # BooleanOptionalAction pair: the ONLY spelling of the negative
        # sense is the --no-* alias, and it takes no value (nargs 0).
        "enable-prefix-caching": {"type": None, "choices": None, "nargs": 0,
                                  "widget": "toggle",
                                  "aliases": ["--no-enable-prefix-caching"],
                                  "default": "False", "help": "", "repeatable": False},
        # Append-action repeatable flag: a list of any length is legal,
        # each element validated on its own.
        "served-model-name": {"type": "str", "choices": None, "nargs": None,
                              "widget": "list", "aliases": [], "default": "None",
                              "help": "", "repeatable": True},
        "gpu-ids": {"type": "int", "choices": None, "nargs": None, "widget": "list",
                    "aliases": [], "default": "None", "help": "", "repeatable": True},
        # nargs +/*: also a legal list shape without being "repeatable".
        "stop": {"type": "str", "choices": None, "nargs": "+", "widget": "list",
                 "aliases": [], "default": "None", "help": "", "repeatable": False},
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
    # F5 regression: entry missing "value" entirely.
    assert isinstance(validate_settings({"foo": {}}, CATALOG, {}), list)
    # F5 regression: entry isn't a dict at all.
    assert isinstance(validate_settings({"foo": "bar"}, CATALOG, {}), list)
    # F5 regression: raw int value leaking through the normalization
    # invariant (resolved values are supposed to be str/True/list[str]).
    assert isinstance(
        validate_settings(
            {"max-model-len": {"value": 5, "origin": "declared", "layer": "engine"}},
            CATALOG, {},
        ),
        list,
    )


# --- F1: catalog aliases (BooleanOptionalAction --no-* halves, short forms) ---

def test_negative_alias_of_a_boolean_optional_flag_is_not_unknown():
    """`--no-enable-prefix-caching` is the ONLY spelling of the negative
    sense — it must resolve through the catalog's aliases, not miss and
    read as unknown."""
    warnings = validate_settings(
        _resolved(**{"no-enable-prefix-caching": True}), CATALOG, {})

    assert warnings == []


def test_toggle_option_given_a_value_warns():
    """nargs 0 means the engine takes no value for this flag — the catalog
    carries nargs, so this is now catchable (M2)."""
    warnings = validate_settings(
        _resolved(**{"enable-prefix-caching": "yes"}), CATALOG, {})

    assert warnings[0]["class"] == "type"


def test_short_alias_resolves_to_the_canonical_option_for_type_checking():
    """`-q` is quantization's short form. It must resolve to the
    canonical option entry for type/choices checking, and the warning
    must still be keyed by the settings key the caller actually used."""
    warnings = validate_settings(_resolved(q="banana"), CATALOG, {})

    assert warnings[0]["class"] == "type"
    assert warnings[0]["key"] == "q"
    assert "modelopt" in warnings[0]["message"]


# --- F4: list-valued settings ---

def test_list_on_a_repeatable_option_with_valid_elements_is_silent():
    warnings = validate_settings(
        _resolved(**{"served-model-name": ["a", "b", "c"]}), CATALOG, {})

    assert warnings == []


def test_list_on_a_repeatable_option_flags_the_bad_element():
    warnings = validate_settings(
        _resolved(**{"gpu-ids": ["0", "not-a-number", "2"]}), CATALOG, {})

    assert len(warnings) == 1
    assert warnings[0]["class"] == "type"
    assert warnings[0]["key"] == "gpu-ids"


def test_list_on_a_non_repeatable_single_value_option_warns():
    """max-model-len takes exactly one value; a list is illegal for it
    regardless of what the elements look like."""
    warnings = validate_settings(
        _resolved(**{"max-model-len": ["131072", "262144"]}), CATALOG, {})

    assert len(warnings) == 1
    assert warnings[0]["class"] == "type"
    assert warnings[0]["key"] == "max-model-len"


def test_list_on_an_nargs_plus_option_is_legal():
    warnings = validate_settings(_resolved(**{"stop": ["a", "b"]}), CATALOG, {})

    assert warnings == []


# --- F2/F3: the dash<->underscore conflict bridge ---

def test_bridged_conflict_is_keyed_by_the_dash_form_settings_key():
    """detect_drift speaks fact-field vocabulary (underscored); settings
    are always dash-form. THE regression: the bridge must be mechanical
    and reversible, not a hand-listed special case, and the warning must
    come back keyed the way every settings-key consumer looks it up."""
    warnings = validate_settings(
        _resolved(**{"max-model-len": "200000"}),
        CATALOG,
        _facts(max_position_embeddings=131072),
    )

    conflicts = [w for w in warnings if w["class"] == "conflict"]
    assert len(conflicts) == 1
    assert conflicts[0]["key"] == "max-model-len"
    assert conflicts[0]["severity"] == "mismatch"


def test_max_input_tokens_conflict_is_reachable_from_settings():
    """max_input_tokens is a real DRIFT_RULES field that the original
    hand-added bridge (quantization, max-model-len only) could never
    reach. The mechanical bridge must reach it without a new special
    case."""
    warnings = validate_settings(
        _resolved(**{"max-input-tokens": "200000"}),
        CATALOG,
        _facts(max_model_len_live="131072"),
    )

    conflicts = [w for w in warnings if w["class"] == "conflict"]
    assert len(conflicts) == 1
    assert conflicts[0]["key"] == "max-input-tokens"
    assert conflicts[0]["severity"] == "mismatch"
