"""Tests for app.harvest — engine option catalogs.

The recipe was validated live against sparky's vLLM image on 2026-08-04:
argparse INTROSPECTION, not --help scraping (which loses every type and
choice), executed inside the RUNNING container (--help crashes without a
GPU). These tests run against captured probe output — no container needed.
"""

import json

from app.harvest import parse_probe_output, widget_for

PROBE_OUTPUT = json.dumps({
    "options": [
        {"flags": ["--max-model-len"], "type": "int", "choices": None,
         "default": "None", "nargs": None, "cls": "_StoreAction",
         "help": "Model context length."},
        {"flags": ["--quantization", "-q"], "type": "str",
         "choices": ["modelopt", "compressed-tensors", "fp8"],
         "default": "None", "nargs": None, "cls": "_StoreAction",
         "help": "Method used to quantize the weights."},
        {"flags": ["--enable-prefix-caching"], "type": None, "choices": None,
         "default": "False", "nargs": 0, "cls": "_StoreTrueAction",
         "help": "Enable prefix caching."},
        {"flags": ["--served-model-name"], "type": "str", "choices": None,
         "default": "None", "nargs": "+", "cls": "_StoreAction",
         "help": "Model name(s) used in the API."},
        {"flags": ["-h", "--help"], "type": None, "choices": None,
         "default": "==SUPPRESS==", "nargs": 0, "cls": "_HelpAction",
         "help": "show this help message and exit"},
    ]
})


def test_catalog_keyed_by_long_flag_without_dashes():
    catalog = parse_probe_output(PROBE_OUTPUT, engine_version="0.26.0", now="t")

    assert "max-model-len" in catalog["value"]["options"]


def test_help_action_is_excluded():
    """-h is not a setting."""
    catalog = parse_probe_output(PROBE_OUTPUT, engine_version="0.26.0", now="t")

    assert "help" not in catalog["value"]["options"]


def test_type_and_choices_are_captured():
    options = parse_probe_output(PROBE_OUTPUT, "0.26.0", "t")["value"]["options"]

    assert options["quantization"]["type"] == "str"
    assert options["quantization"]["choices"] == ["modelopt", "compressed-tensors", "fp8"]


def test_short_alias_is_recorded():
    options = parse_probe_output(PROBE_OUTPUT, "0.26.0", "t")["value"]["options"]

    assert options["quantization"]["aliases"] == ["-q"]


def test_engine_version_is_stamped_on_the_catalog():
    """A catalog is only valid for the version it was harvested from —
    without this the UI cannot know it is stale."""
    catalog = parse_probe_output(PROBE_OUTPUT, engine_version="0.26.0", now="t")

    assert catalog["value"]["engine_version"] == "0.26.0"
    assert catalog["source"] == "argparse introspection"
    assert catalog["derived_ts"] == "t"


def test_malformed_probe_output_yields_no_catalog():
    """A probe that failed must produce nothing, not a half catalog that
    looks authoritative."""
    assert parse_probe_output("not json", "0.26.0", "t") == {}


def test_widget_toggle_for_bare_boolean():
    assert widget_for({"nargs": 0, "type": None, "choices": None}) == "toggle"


def test_widget_select_for_choices():
    assert widget_for({"nargs": None, "type": "str", "choices": ["a", "b"]}) == "select"


def test_widget_list_for_multi_value():
    assert widget_for({"nargs": "+", "type": "str", "choices": None}) == "list"


def test_widget_number_for_int():
    assert widget_for({"nargs": None, "type": "int", "choices": None}) == "number"


def test_widget_text_is_the_fallback():
    """An unknown action class degrades to free text rather than erroring —
    a flag we cannot render a widget for must still be editable."""
    assert widget_for({"nargs": None, "type": "_optional_type", "choices": None}) == "text"


def test_probe_source_uses_python3_not_python():
    """The image has no `python` on PATH — validated live 2026-08-04."""
    from app.harvest import PROBE_INTERPRETER

    assert PROBE_INTERPRETER == "python3"
