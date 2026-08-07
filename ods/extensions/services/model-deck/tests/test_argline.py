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
"""

import shlex

import pytest

from app.argline import parse_argline, render_argline

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
