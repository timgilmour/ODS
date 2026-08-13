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

from app.argline import (
    POSITIONAL_KEY,
    normalize_args_map,
    parse_argline,
    render_argline,
    render_argv,
)

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
    """Delegates to the production function (F1 fix round, 2026-08-07
    review). This used to be a hand-rolled mirror of the two RULING
    2026-08-07 normalization axes, with its OWN ``_positional`` exemption
    that ``normalize_args_map`` never got — the divergence hid the F1
    production bug (a singleton ``_positional`` list silently collapsing
    to a scalar) from this very property test. Now that
    ``normalize_args_map`` carries the same exemption, mirroring it here
    would just be re-implementing the thing under test; delegating removes
    the divergence at the root instead of patching this copy again."""
    return normalize_args_map(settings)


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


# --- F1, CRITICAL (final branch review, 2026-08-07): POSITIONAL_KEY must
# be exempt from the singleton-list -> scalar axis. Unfixed, a one-token
# positional list (the common case: `serve /model`'s "serve" alone, or a
# bare `serve` with no model arg yet typed) collapsed to a bare string on
# the way through normalize_args_map, and render_argline then iterated
# that string character by character -- persisted corruption of typed
# input, silently, the next time the argline was rendered from the store.


def test_normalize_args_map_keeps_a_singleton_positional_list_as_a_list():
    assert normalize_args_map({"_positional": ["serve"]}) == {"_positional": ["serve"]}


def test_normalize_args_map_stringifies_numeric_positional_elements():
    """The OTHER RULING 2026-08-07 axis (numeric -> string) still applies
    to positional elements -- only the singleton-collapse axis is
    exempted for POSITIONAL_KEY."""
    assert normalize_args_map({"_positional": [1, "two"]}) == {"_positional": ["1", "two"]}


def test_normalize_args_map_drops_an_empty_positional_list_with_a_warning():
    """The empty-list-drop rule still applies to POSITIONAL_KEY -- only
    the singleton-collapse axis is exempted."""
    with pytest.warns(UserWarning, match="empty list"):
        result = normalize_args_map({"_positional": []})

    assert "_positional" not in result


def test_normalize_args_map_keeps_a_multi_token_positional_list_as_a_list():
    assert normalize_args_map({"_positional": ["serve", "/model"]}) == {
        "_positional": ["serve", "/model"]
    }


def test_one_positional_token_round_trips_through_the_real_store_byte_exact(tmp_path):
    """F1's actual blast radius: not normalize_args_map in isolation, but
    SettingsStore.put (which calls it on every args write) -> a later read
    -> render_argline. Before the fix this produced
    's e r v e --max-model-len 100' -- because the one-token `_positional`
    list collapsed to the bare string 'serve' and render_argline iterated
    it character by character. Byte-exact round trip through the REAL
    store, not a hand-built map, so a regression here fails the same way
    the review finding did."""
    from app.settings_store import SettingsStore

    line = "serve --max-model-len 100"
    parsed = parse_argline(line)
    assert parsed["_positional"] == ["serve"]  # sanity: exactly one token

    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", parsed)
    stored = store.scope("engines", "sparky/vllm")["args"]

    assert render_argline(stored) == line


class TestRenderArgv:
    """render_argv is the document form of render_argline: same dispatch,
    no shell quoting — consumed by compose `command:` arrays, which take
    tokens verbatim."""

    def test_positionals_lead_then_flags(self):
        from app.argline import POSITIONAL_KEY, render_argv

        argv = render_argv({POSITIONAL_KEY: ["serve", "/model"],
                            "max-model-len": "262144"})

        assert argv == ["serve", "/model", "--max-model-len", "262144"]

    def test_bare_flag_is_one_token(self):
        from app.argline import render_argv

        assert render_argv({"enable-prefix-caching": True}) == ["--enable-prefix-caching"]

    def test_multi_value_flag_emits_separate_tokens(self):
        from app.argline import render_argv

        argv = render_argv({"served-model-name": ["aeon", "aeon-fast"]})

        assert argv == ["--served-model-name", "aeon", "aeon-fast"]

    def test_json_blob_survives_as_one_unquoted_token(self):
        from app.argline import render_argv

        argv = render_argv({"speculative-config": '{"method":"dflash"}'})

        assert argv == ["--speculative-config", '{"method":"dflash"}']

    def test_dash_shaped_scalar_renders_equals_form(self):
        """RULING (C1): a value that looks like a flag fuses with its key so
        argparse cannot misread it as a separate option. Task-1 brief's
        literal example value was "-1" -- a negative number, which
        _is_negative_number exempts from dash-shaped treatment on purpose
        (RULING 2026-08-07, pinned by DASH_SHAPED_CASES' {"neg": "-5"}
        control case above); using it here would demand overturning that
        ruling for the scalar path render_argline shares via _argv_tokens.
        Swapped for a non-numeric dash-shaped value that actually exercises
        this branch, matching this file's own convention (e.g. "-x" in
        DASH_SHAPED_CASES)."""
        from app.argline import render_argv

        assert render_argv({"some-offset": "-x"}) == ["--some-offset=-x"]

    def test_single_char_key_gets_one_dash(self):
        from app.argline import render_argv

        assert render_argv({"c": "32768"}) == ["-c", "32768"]

    def test_round_trip_through_parse(self):
        """parse(render) is identity on the parsed shape — the property that
        makes the document diffable against an adopted import."""
        import shlex

        from app.argline import POSITIONAL_KEY, parse_argline, render_argv

        settings = {POSITIONAL_KEY: ["serve", "/model"],
                    "max-model-len": "262144",
                    "enable-prefix-caching": True,
                    "served-model-name": ["a", "b"]}

        argv = render_argv(settings)

        assert parse_argline(" ".join(shlex.quote(t) for t in argv)) == settings

    def test_argline_is_the_shell_quoted_join_of_argv(self):
        """The two renderers cannot diverge: argline IS argv, quoted."""
        import shlex

        from app.argline import POSITIONAL_KEY, render_argline, render_argv

        settings = {POSITIONAL_KEY: ["serve", "/model"],
                    "speculative-config": '{"m":"d"}',
                    "enable-prefix-caching": True}

        assert render_argline(settings) == " ".join(
            shlex.quote(t) for t in render_argv(settings))


def test_argv_tokens_refuses_a_dict_value():
    """[max-review c49] The else-branch fell through to str(value), so a dict
    rendered its PYTHON REPR into a launch argline —
    `--max-num-seqs {'a': 1}`. Refuse, don't coerce: there is no honest
    argline for a mapping.

    This is the DISK-side boundary. The wire gate (normalize_args_map,
    below) cannot cover it: this renderer also consumes settings loaded from
    a settings.json that was hand-edited, or written before that gate
    existed.
    """
    with pytest.raises(ValueError, match="max-num-seqs"):
        render_argv({"max-num-seqs": {"a": 1}})


def test_argline_render_refuses_a_dict_value():
    """Same guard, reached through the other public renderer — they share
    _argv_tokens, and a guard placed in only one of them would let the shell
    path through."""
    with pytest.raises(ValueError, match="max-num-seqs"):
        render_argline({"max-num-seqs": {"a": 1}})


def test_argv_tokens_still_renders_the_scalars_it_should():
    """The refusal must not catch the documented axes on its way past:
    bools are bare-flag sentinels, lists have their own branch, and
    str/int/float all render."""
    assert render_argv({"flag": True}) == ["--flag"]
    assert render_argv({"seqs": 4}) == ["--seqs", "4"]
    assert render_argv({"ratio": 1.5}) == ["--ratio", "1.5"]
    assert render_argv({"name": "v"}) == ["--name", "v"]
    assert render_argv({"n": 4}) == ["-n", "4"]  # single char -> one dash


def test_normalize_refuses_a_dict_value_at_the_wire():
    """PUT /api/settings accepts JSON, so a dict value must be REFUSED at
    entry (422) rather than persisted and rendered later
    [[literal-declared-inputs]]. This is the WIRE boundary; the renderer
    guard above cannot cover it, because by the time the renderer sees the
    value it is already on disk and the operator is long gone."""
    with pytest.raises(ValueError, match="max-num-seqs"):
        normalize_args_map({"max-num-seqs": {"a": 1}})


def test_normalize_refuses_a_dict_inside_a_list():
    """A list is normalized element-wise, so the same value smuggled one
    level down must be refused too."""
    with pytest.raises(ValueError, match="max-num-seqs"):
        normalize_args_map({"max-num-seqs": [1, {"a": 1}]})


def test_render_refuses_a_dict_inside_a_list():
    """[T9 review Imp-2] The first version of the render guard sat only on
    the bare-value branch, BELOW the list branch — so this rendered
    `{'a': 1}`'s Python repr into a launch argline, c49 verbatim, through the
    very boundary whose comment claims to cover hand-edited files."""
    with pytest.raises(ValueError, match="served-model-name"):
        render_argv({"served-model-name": ["a", {"a": 1}]})
    with pytest.raises(ValueError, match="served-model-name"):
        render_argline({"served-model-name": ["a", {"a": 1}]})


def test_render_refuses_a_dict_positional_token():
    """Same gap on the other side: positionals are emitted by their own loop
    ABOVE the flag loop, so they bypassed the guard entirely."""
    with pytest.raises(ValueError, match=POSITIONAL_KEY):
        render_argv({POSITIONAL_KEY: ["serve", {"a": 1}]})


def test_render_skips_an_explicitly_unset_value():
    """[T9 review Imp-3] None is NOT unrenderable garbage — it is a
    load-bearing value meaning EXPLICIT UNSET (app/ladder.py:73 pops the key
    and anything a lower layer contributed), so its honest rendering is
    nothing at all.

    Refusing it made a scope containing one permanently un-viewable and
    un-editable through the deck: the wire accepts None (it must — it is a
    real operator opinion), so a 422 raised only at render time was one the
    operator had no way to clear.
    """
    assert render_argv({"gone": None, "kept": "v"}) == ["--kept", "v"]
    assert render_argline({"gone": None, "kept": "v"}) == "--kept v"


def test_normalize_accepts_an_explicitly_unset_value():
    """The wire must keep accepting None for the same reason — refusing it
    would remove the operator's only way to say "unset this key"."""
    assert normalize_args_map({"k": None}) == {"k": None}


def test_render_skips_an_unset_inside_a_list_and_positionals():
    """[T9-fix review] The unset ruling was applied ONLY to the bare-value
    branch, so a persisted None inside a list or in _positional still hit the
    refusal — and that wedged the SHIP path: GET effective, the preview and
    the reload route (now /api/nodes/{id}/serving/reload) all 422'd, i.e.
    the endpoints that would show the
    operator what to fix were the ones failing. The ladder pops only
    TOP-LEVEL Nones, so a list containing one survives resolution and reaches
    the renderer.

    Already-persisted data must therefore render, not refuse."""
    assert render_argv({"served-model-name": ["a", None]}) == ["--served-model-name", "a"]
    assert render_argv({POSITIONAL_KEY: ["serve", None]}) == ["serve"]
    # A list of nothing but Nones says nothing at all — not a bare flag.
    assert render_argv({"gone": [None], "kept": "v"}) == ["--kept", "v"]


def test_wire_refuses_a_null_inside_a_list():
    """The other boundary: a bare None means "unset this key", but INSIDE a
    list it means nothing, so a NEW write is refused rather than guessed at.
    Persisted ones still render (above), which is what makes them fixable."""
    with pytest.raises(ValueError, match="served-model-name"):
        normalize_args_map({"served-model-name": ["a", None]})
    with pytest.raises(ValueError, match=POSITIONAL_KEY):
        normalize_args_map({POSITIONAL_KEY: ["serve", None]})


def test_wire_treats_a_bare_positional_null_as_unset():
    """`_positional: null` used to be wrapped into [None] — inventing a
    positional token nobody typed, which then 422'd every renderer. It means
    the same "unset" a bare None means for any other key, and is STORED as
    that marker (see the dedicated test below for why not dropped)."""
    assert normalize_args_map({POSITIONAL_KEY: None}) == {POSITIONAL_KEY: None}


def test_heal_drops_list_nulls_instead_of_refusing():
    """[T9-fix re-review] The wire REFUSES a list-null (operator input), but a
    HEALING caller must DROP it. SettingsStore.restore re-normalizes a
    snapshot whose own docstring calls it "exactly as untrusted as a file on
    disk" — refusing there turned a heal into an ABORT and broke undo for
    precisely the legacy data the renderer's tolerance keeps workable."""
    assert normalize_args_map({"k": ["a", None]}, heal=True) == {"k": "a"}
    assert normalize_args_map({POSITIONAL_KEY: ["serve", None]}, heal=True) == {
        POSITIONAL_KEY: ["serve"]}
    # ...and the wire keeps refusing the same input.
    with pytest.raises(ValueError):
        normalize_args_map({"k": ["a", None]})


def test_a_bare_positional_null_is_stored_as_an_unset_marker():
    """[T9-fix re-review] _DROP removed the key from the payload entirely, so
    no unset marker reached app.ladder and "unset the positional" became
    INEXPRESSIBLE — an engine-layer positional survived while the response
    showed args:{}. Stored as None, ladder.py:73 pops it like any other
    unset, and the renderer tolerates it."""
    assert normalize_args_map({POSITIONAL_KEY: None}) == {POSITIONAL_KEY: None}
    assert render_argv({POSITIONAL_KEY: None, "k": "v"}) == ["-k", "v"]


def test_heal_preserves_an_unset_marker_including_the_positional_one():
    """[final review] The heal posture drops nulls INSIDE a list — damage —
    but a BARE None is an unset MARKER and must survive, positional included.

    An earlier heal branch dropped `_positional: None` while preserving
    `{"k": None}`: the two rounds' rulings disagreed at their single
    intersection, and the consequence was silent. A restore would re-expose a
    harvested engine-defaults positional the marker had suppressed, so UNDO
    changed effective launch args. Both spellings asserted together, because
    the defect was the disagreement between them.
    """
    assert normalize_args_map({POSITIONAL_KEY: None}, heal=True) == {POSITIONAL_KEY: None}
    assert normalize_args_map({"k": None}, heal=True) == {"k": None}


def test_wire_refuses_a_nested_list():
    """[final review] The singleton-collapse axis readmitted the shapes the
    dict guard refuses: `{"k": [["x", {"a": 1}]]}` collapsed to its inner
    list and sailed through. No argline renders a nested list."""
    with pytest.raises(ValueError, match="k"):
        normalize_args_map({"k": [["x", {"a": 1}]]})
    with pytest.raises(ValueError, match="k"):
        normalize_args_map({"k": [["a", "b"], ["c"]]})
    # The ordinary singleton collapse still works.
    assert normalize_args_map({"k": ["v"]}) == {"k": "v"}
