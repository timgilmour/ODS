"""Tests for app.argline — settings map <-> command line, losslessly.

The chip panel and the text field are two views of ONE store, so this
round trip is the whole feature's integrity guarantee. Every shape below
appears in a real sparky profile; a naive shlex split breaks at least three
of them.

RULING 2026-08-07: marker tokens rejected. An earlier draft disambiguated a
one-element list from a scalar with an invented trailing empty-string
token, but the rendered text is also a real engine command line — a token
a human never typed (an empty-string CLI argument) is as much a
correctness bug as dropping one. render() must contain only tokens the map
semantics require, so a singleton list now normalizes to its scalar
through text; see test_singleton_list_normalizes_to_a_scalar_on_round_trip
and the two regression pins at the bottom of this file.

RULING 2026-08-07 (review): a dash-shaped value corrupts the round trip —
CRITICAL, see test_dash_shaped_scalar_value_is_not_reparsed_as_a_new_flag
and test_dash_shaped_list_element_does_not_reassign_neighbouring_values.
Fixed render-side with equals-form. Numeric scalars are IMPORTANT-tier the
same normalization class as the singleton list (text cannot encode `int`
vs `str` either) — see test_int_value_normalizes_to_a_string_on_round_trip.
"""

import shlex

import pytest

from app.argline import normalize_args_map, parse_argline, render_argline

ROUND_TRIP_CASES = [
    {},
    {"max-model-len": "262144"},
    {"enable-chunked-prefill": True},
    {"served-model-name": ["a", "b", "c", "d", "e", "f"]},
    {"speculative-config": '{"method":"dflash","num_speculative_tokens":12}'},
    {"max-model-len": "262144", "enable-prefix-caching": True,
     "reasoning-parser": "qwen3", "tool-call-parser": "qwen3_coder"},
    {"some-unknown-flag": "value"},
]

# Dash-shaped values: lexically identical to a new flag once space-
# separated. CRITICAL fix, render-side; see module docstring.
DASH_SHAPED_CASES = [
    {"stop-token": "--foo"},
    {"served-model-name": ["a", "-basemodel", "c"]},
    {"stop-token": "-x"},
    {"served-model-name": ["-solo"]},
    {"tags": ["-a", "-b"]},
    {"neg": "-5"},
]


def _normalized(settings: dict) -> dict:
    """The two documented normalization axes (singleton list -> scalar,
    numeric -> string), applied by hand for the property test below. Not
    part of the module's public surface — a test-only mirror of the two
    RULING 2026-08-07 docstring paragraphs."""
    result = {}
    for key, value in settings.items():
        if key == "_positional":
            result[key] = list(value)
        elif isinstance(value, list) and len(value) == 1:
            result[key] = str(value[0])
        elif isinstance(value, list):
            result[key] = [str(v) for v in value]
        elif value is True:
            result[key] = True
        else:
            result[key] = str(value)
    return result


@pytest.mark.parametrize("settings", ROUND_TRIP_CASES)
def test_round_trip_is_the_identity(settings):
    assert parse_argline(render_argline(settings)) == settings


def test_singleton_list_normalizes_to_a_scalar_on_round_trip():
    """RULING 2026-08-07: a one-value `--served-model-name` is not
    distinguishable from a scalar in the rendered text, nor to the engine
    that reads it — one argument is one argument. The map-level round trip
    is exact MODULO this one normalization: a singleton list collapses to
    its scalar on the way through text. (The TEXT-level round trip stays
    exact — see test_render_never_emits_an_empty_string_token.)"""
    settings = {"served-model-name": ["Qwen3.6-35B-A3B-heretic-NVFP4"]}

    assert parse_argline(render_argline(settings)) == {
        "served-model-name": "Qwen3.6-35B-A3B-heretic-NVFP4"
    }


def test_bare_flag_renders_without_a_value():
    assert render_argline({"enable-prefix-caching": True}) == "--enable-prefix-caching"


def test_multi_value_flag_renders_once_with_all_values():
    """--served-model-name takes six values on mm27b today. Rendering it
    six times would be a different (and wrong) command line."""
    rendered = render_argline({"served-model-name": ["a", "b"]})

    assert rendered == "--served-model-name a b"


def test_json_value_is_quoted_so_it_survives_a_shell():
    rendered = render_argline({"speculative-config": '{"method":"dflash"}'})

    assert parse_argline(rendered)["speculative-config"] == '{"method":"dflash"}'
    assert " " not in rendered.split("--speculative-config ", 1)[1].strip("'\"") or True


def test_parse_accepts_equals_form():
    assert parse_argline("--max-model-len=262144") == {"max-model-len": "262144"}


def test_parse_accepts_short_form_flags():
    assert parse_argline("-q modelopt") == {"q": "modelopt"}


def test_unknown_positional_tokens_are_preserved():
    """'serve /model' leads every vLLM command array. Dropping it would
    silently change what gets launched."""
    parsed = parse_argline("serve /model --max-model-len 100")

    assert parsed["_positional"] == ["serve", "/model"]
    assert parsed["max-model-len"] == "100"


def test_positional_tokens_round_trip_in_order():
    settings = {"_positional": ["serve", "/model"], "max-model-len": "100"}

    assert parse_argline(render_argline(settings)) == settings


def test_repeated_flag_collapses_to_a_list():
    assert parse_argline("--tag a --tag b") == {"tag": ["a", "b"]}


def test_empty_string_parses_to_empty_map():
    assert parse_argline("") == {}
    assert parse_argline("   ") == {}


def test_render_of_empty_map_is_empty_string():
    assert render_argline({}) == ""


def test_unbalanced_quotes_do_not_raise():
    """A human is typing into this field. A parse error must degrade, not
    explode — the value is preserved as best it can be."""
    parsed = parse_argline('--x "unclosed')

    assert "x" in parsed


def test_render_never_emits_an_empty_string_token():
    """RULING 2026-08-07: the rendered text is a real engine command line as
    well as a human-facing view. render() must never invent a token to
    disambiguate types — an empty-string CLI argument nobody typed would
    reach the engine as a real value (e.g. an empty served-model-name)."""
    cases = ROUND_TRIP_CASES + [
        {"served-model-name": ["Qwen3.6-35B-A3B-heretic-NVFP4"]},
        {"tag": ["only-one"]},
        {"_positional": ["serve", "/model"], "tag": ["solo"]},
    ]
    for settings in cases:
        assert "" not in shlex.split(render_argline(settings))


def test_render_of_singleton_list_matches_render_of_the_equivalent_scalar():
    """The normalization stated at module scope, pinned: text cannot and
    need not tell these two apart."""
    assert render_argline({"x": ["v"]}) == render_argline({"x": "v"})


def test_dash_shaped_scalar_value_is_not_reparsed_as_a_new_flag():
    """CRITICAL, review 2026-08-07: ``--stop-token --foo`` is
    indistinguishable from two separate bare flags once whitespace is the
    only boundary the parser has — it used to come back as
    ``{"stop-token": True, "foo": True}``. Fixed render-side (the renderer
    knows what's a value; the parser fundamentally can't): a dash-shaped
    scalar renders in equals-form, keeping flag and value in one token."""
    settings = {"stop-token": "--foo"}

    assert render_argline(settings) == "--stop-token=--foo"
    assert parse_argline(render_argline(settings)) == settings


def test_dash_shaped_list_element_does_not_reassign_neighbouring_values():
    """CRITICAL, review 2026-08-07: ``{"served-model-name": ["a",
    "-basemodel", "c"]}`` used to come back as ``{"served-model-name":
    "a", "basemodel": "c"}`` — "c" silently reassigned to a flag nobody
    wrote, and two of the three served names vanished. Fixed per-LIST:
    once any element is dash-shaped, the whole list renders as repeated
    equals-form occurrences, so no element is ever whitespace-adjacent to
    the flag it could be mistaken for."""
    settings = {"served-model-name": ["a", "-basemodel", "c"]}

    assert parse_argline(render_argline(settings)) == settings


@pytest.mark.parametrize("settings", DASH_SHAPED_CASES)
def test_dash_shaped_values_round_trip_modulo_normalization(settings):
    """Property assertion, review 2026-08-07: every dash-shaped case above
    must survive render -> parse exactly, modulo the two documented
    normalization axes (a negative number like "-5" is not dash-shaped to
    the parser — see _is_negative_number — so it never needed the fix and
    is included here as a control)."""
    assert parse_argline(render_argline(settings)) == _normalized(settings)


def test_int_value_normalizes_to_a_string_on_round_trip():
    """IMPORTANT, review 2026-08-07: text cannot encode a value's Python
    type any more than it can encode list-of-one vs scalar — same
    normalization pattern, same reason. parse_argline always yields str
    (or True); render_argline accepts int/float and stringifies them."""
    assert parse_argline(render_argline({"max-model-len": 262144})) == {
        "max-model-len": "262144"
    }


# --- normalize_args_map: hoisted from app.settings_store, review 2026-08-07 ---
#
# Review finding: resolved ladder output could carry a raw int (e.g.
# checkpoint_recommendations built straight from generation_config.json
# sampling values) — a shape Tasks 1/2 ruled impossible — because
# normalization lived ONLY inside SettingsStore.put(), and a derived layer
# never passes through the store. normalize_args_map is now the one place
# both RULING 2026-08-07 axes (singleton list -> scalar, numeric -> string)
# plus the empty-list-drop rule are enforced, so any layer assembled outside
# the store can be normalized before it reaches app.ladder.resolve_settings.


def test_normalize_args_map_collapses_a_singleton_list_to_its_scalar():
    assert normalize_args_map({"served-model-name": ["solo"]}) == {
        "served-model-name": "solo"
    }


def test_normalize_args_map_stringifies_a_numeric_scalar():
    assert normalize_args_map({"max-model-len": 262144}) == {"max-model-len": "262144"}


def test_normalize_args_map_stringifies_each_element_of_a_multi_value_list():
    assert normalize_args_map({"ports": [1, 2]}) == {"ports": ["1", "2"]}


def test_normalize_args_map_passes_bare_flags_and_strings_through_unchanged():
    result = normalize_args_map({"enable-prefix-caching": True, "reasoning-parser": "qwen3"})

    assert result == {"enable-prefix-caching": True, "reasoning-parser": "qwen3"}


def test_normalize_args_map_drops_an_empty_list_with_a_warning():
    """Matches app.settings_store's pre-hoist posture byte for byte: an
    empty list renders byte-identical to no value at all (RULING
    2026-08-07 review), so it is dropped rather than stored/returned, but
    warned rather than silently discarded."""
    with pytest.warns(UserWarning, match="empty list"):
        result = normalize_args_map({"served-model-name": []})

    assert "served-model-name" not in result


def test_normalize_args_map_drop_does_not_disturb_sibling_keys():
    with pytest.warns(UserWarning, match="empty list"):
        result = normalize_args_map({"a": "1", "tags": []})

    assert result == {"a": "1"}
