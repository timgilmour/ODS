"""Harvest an engine's legal-option catalog from the engine itself.

Recipe validated live against sparky's vLLM image, 2026-08-04. Two findings
shaped it, and both contradict the obvious approach:

1. **``vllm serve --help`` crashes without a GPU** — it builds VllmConfig
   defaults, which call device inference (``RuntimeError: Failed to infer
   device type``). Harvest therefore runs INSIDE THE RUNNING engine
   container, where a GPU is visible. A consequence worth stating plainly:
   an engine that has never been up has no catalog, and that is a supported
   state, not an error (validation warns rather than blocks, so every flag
   stays editable without one).

2. **Text-scraping ``--help`` loses the machine-readable half.** Modern
   vLLM's bare ``--help`` lists only config groups; ``--help=all`` is 1821
   lines and 281 flags, but types and choices appear only as English prose
   — the string "Possible choices" appears ZERO times. Argparse
   INTROSPECTION instead yields 275 options, 194 with a concrete type, 32
   with explicit choices, plus defaults, aliases, nargs and action classes.
   That is a schema; the help text is not.

The probe must be run with ``python3``: the image has no ``python`` on PATH.

Parsing is pure and lives here; execution (docker exec) belongs to the
caller, which keeps this module testable against captured output.

The catalog's ``options[*].default`` values are raw argparse defaults, not
argline-normalized ones. A caller assembling an engine_defaults layer from
them (Task 7) must pass them through ``app.argline.normalize_args_map``
first, same as any other args-shaped layer assembled outside
app.settings_store.
"""

import json

PROBE_INTERPRETER = "python3"

# Executed inside the running engine container. Prints one JSON object.
PROBE_SOURCE = '''
import json
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser

parser = make_arg_parser(FlexibleArgumentParser())
options = []
for action in parser._actions:
    if not action.option_strings:
        continue
    t = getattr(action, "type", None)
    options.append({
        "flags": action.option_strings,
        "type": getattr(t, "__name__", str(t) if t else None),
        "choices": list(action.choices) if action.choices else None,
        "default": repr(action.default),
        "nargs": action.nargs,
        "cls": type(action).__name__,
        "help": action.help or "",
    })
print(json.dumps({"options": options}))
'''

_EXCLUDED_ACTIONS = ("_HelpAction", "_VersionAction")


def parse_probe_output(text: str, engine_version: str, now: str) -> dict:
    """Parse probe JSON into ONE characteristics field holding the catalog.

    Returns ``{}`` for unparseable output: a probe that failed must produce
    nothing rather than a half catalog that looks authoritative.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    options: dict[str, dict] = {}
    for entry in payload.get("options") or []:
        if entry.get("cls") in _EXCLUDED_ACTIONS:
            continue
        flags = entry.get("flags") or []
        long_flags = [f for f in flags if f.startswith("--")]
        if not long_flags:
            continue
        name = long_flags[0].lstrip("-")
        options[name] = {
            "aliases": [f for f in flags if f not in long_flags],
            "type": entry.get("type"),
            "choices": entry.get("choices"),
            "default": entry.get("default"),
            "nargs": entry.get("nargs"),
            "help": entry.get("help", ""),
            "widget": widget_for(entry),
        }

    if not options:
        return {}

    return {
        "value": {"engine_version": engine_version, "options": options},
        "source": "argparse introspection",
        "derived_ts": now,
    }


def widget_for(option: dict) -> str:
    """Pick a control for an option. Unknown shapes degrade to free text —
    a flag we cannot render nicely must still be editable."""
    if option.get("nargs") == 0:
        return "toggle"
    if option.get("choices"):
        return "select"
    if option.get("nargs") in ("+", "*"):
        return "list"
    if option.get("type") in ("int", "float"):
        return "number"
    return "text"
