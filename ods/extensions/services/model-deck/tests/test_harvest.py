"""Tests for app.harvest — engine option catalogs.

The recipe was validated live against sparky's vLLM image on 2026-08-04:
argparse INTROSPECTION, not --help scraping (which loses every type and
choice), executed inside the RUNNING container (--help crashes without a
GPU). These tests run against captured probe output — no container needed.
"""

import json

from app.harvest import _SENTINEL, PROBE_SOURCE, parse_probe_output, widget_for

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


def test_shipped_probe_file_equals_the_constant():
    """node-agent/swap-helper/harvest_probe.py is executed on sparky;
    app.harvest.PROBE_SOURCE is what parse_probe_output was built against.
    They are the same string or the harvest silently rots."""
    from pathlib import Path

    from app.harvest import PROBE_SOURCE

    shipped = (Path(__file__).resolve().parents[2]
               / "node-agent" / "swap-helper" / "harvest_probe.py").read_text()

    assert shipped == PROBE_SOURCE


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


# --- stdout chatter (fix round 1, CRITICAL) ---------------------------------
#
# vLLM's VLLM_LOGGING_STREAM defaults to ext://sys.stdout: import-time
# platform banners land on the same stream the probe prints its JSON to,
# ahead of it, and json.loads fails on the mix -- harvest then silently
# degrades to the supported no-catalog state against the exact engine it
# targets. PROBE_SOURCE defends in two layers: an env var set before the
# vLLM import (engine-specific), and sentinel-bracketed JSON so the parser
# can find the payload inside noise it did not anticipate (any engine).

def test_probe_source_redirects_vllm_logging_to_stderr_before_import():
    lines = [line.strip() for line in PROBE_SOURCE.strip().splitlines() if line.strip()]
    redirect_idx = next(i for i, l in enumerate(lines) if "VLLM_LOGGING_STREAM" in l)
    import_idx = next(i for i, l in enumerate(lines) if l.startswith(("from vllm", "import vllm")))

    assert redirect_idx < import_idx
    assert "ext://sys.stderr" in lines[redirect_idx]


def test_probe_source_wraps_its_json_in_a_sentinel_pair():
    assert PROBE_SOURCE.count(_SENTINEL) == 2


def test_probe_output_with_leading_banner_noise_parses_between_the_sentinels():
    noisy = (
        "INFO 08-04 12:00:00 [vllm] some platform banner\n"
        "WARNING more stdout chatter from engine startup\n"
        + _SENTINEL + "\n"
        + PROBE_OUTPUT + "\n"
        + _SENTINEL + "\n"
    )
    catalog = parse_probe_output(noisy, "0.26.0", "t")

    assert "max-model-len" in catalog["value"]["options"]


def test_probe_output_with_noise_and_no_sentinel_yields_no_catalog():
    """Without a sentinel marking where the JSON starts, banner noise ahead
    of it breaks whole-string parsing -- this is the exact failure the
    sentinel defends against, not a case it silently recovers from."""
    noisy = "INFO 08-04 12:00:00 [vllm] some platform banner\n" + PROBE_OUTPUT

    assert parse_probe_output(noisy, "0.26.0", "t") == {}


# --- alias completeness (fix round 1, IMPORTANT) -----------------------------
#
# BooleanOptionalAction pairs (--enable-log-requests, --no-enable-log-requests
# -- 21+ real vLLM flags) and mixed long/short groups
# (--data-parallel-external-lb, --no-data-parallel-external-lb, -dpe) both
# have a second LONG flag; the old `f not in long_flags` filter dropped it.

def test_boolean_optional_pair_keeps_the_negative_flag_as_an_alias():
    probe = json.dumps({"options": [
        {"flags": ["--enable-log-requests", "--no-enable-log-requests"],
         "type": None, "choices": None, "default": "True", "nargs": 0,
         "cls": "BooleanOptionalAction", "help": "Enable request logging."},
    ]})
    options = parse_probe_output(probe, "0.26.0", "t")["value"]["options"]

    assert options["enable-log-requests"]["aliases"] == ["--no-enable-log-requests"]


def test_mixed_long_and_short_aliases_are_all_kept():
    probe = json.dumps({"options": [
        {"flags": ["--data-parallel-external-lb",
                    "--no-data-parallel-external-lb", "-dpe"],
         "type": None, "choices": None, "default": "False", "nargs": 0,
         "cls": "BooleanOptionalAction", "help": "Use an external LB."},
    ]})
    options = parse_probe_output(probe, "0.26.0", "t")["value"]["options"]

    assert options["data-parallel-external-lb"]["aliases"] == [
        "--no-data-parallel-external-lb", "-dpe",
    ]


# --- repeatable flags (fix round 1, IMPORTANT) -------------------------------
#
# action="append" flags (e.g. --middleware, default: []) accept the same
# flag multiple times to build a list; without a "repeatable" signal they
# fall through to the "text" widget and lose that semantic.

def test_probe_source_flags_append_actions_as_repeatable():
    assert '"repeatable"' in PROBE_SOURCE
    assert "Append" in PROBE_SOURCE


def test_repeatable_option_gets_the_list_widget():
    probe = json.dumps({"options": [
        {"flags": ["--middleware"], "type": "str", "choices": None,
         "default": "[]", "nargs": None, "cls": "_AppendAction",
         "repeatable": True, "help": "Additional ASGI middleware."},
    ]})
    options = parse_probe_output(probe, "0.26.0", "t")["value"]["options"]

    assert options["middleware"]["repeatable"] is True
    assert options["middleware"]["widget"] == "list"


def test_repeatable_defaults_to_false_when_the_probe_omits_it():
    """Older probe output without the field must not break -- absence
    means non-repeatable, not unknown."""
    options = parse_probe_output(PROBE_OUTPUT, "0.26.0", "t")["value"]["options"]

    assert options["max-model-len"]["repeatable"] is False


def test_widget_list_for_repeatable_flag():
    assert widget_for({"nargs": None, "type": "str", "choices": None, "repeatable": True}) == "list"


# --- malformed option entries (fix round 1, MINOR) ---------------------------

def test_non_dict_option_entry_is_skipped_not_raised():
    """A probe emitting a malformed entry (e.g. a bare string) must not
    crash harvest -- matches the total-degrade posture for bad input."""
    probe = json.dumps({"options": [
        "not-a-dict",
        {"flags": ["--max-model-len"], "type": "int", "choices": None,
         "default": "None", "nargs": None, "cls": "_StoreAction",
         "help": "Model context length."},
    ]})
    options = parse_probe_output(probe, "0.26.0", "t")["value"]["options"]

    assert list(options) == ["max-model-len"]
