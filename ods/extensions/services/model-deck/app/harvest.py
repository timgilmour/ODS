"""Harvest an engine's legal-option catalog from the engine itself.

Recipe validated live against sparky's vLLM image, 2026-08-04. Three
findings shaped it, and all three contradict the obvious approach:

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

3. **Introspection still shares stdout with the engine's own logging.**
   vLLM's ``VLLM_LOGGING_STREAM`` defaults to ``ext://sys.stdout``:
   import-time platform banners land on the SAME stream the probe prints
   its JSON to, ahead of it, and ``json.loads`` fails on the mix — harvest
   then silently degrades to the supported no-catalog state against the
   exact engine it targets, which is worse than an error would be.
   PROBE_SOURCE defends in two layers: it redirects vLLM's own logging to
   stderr before importing vLLM (engine-specific), and it brackets its
   JSON with ``_SENTINEL`` lines so the parser can find the payload inside
   stdout chatter it did not anticipate (any engine).

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

# Marks the JSON payload's boundaries in the probe's stdout so the parser
# can find it inside chatter it did not anticipate (see module docstring,
# finding 3). Must appear nowhere else in PROBE_SOURCE's own output.
_SENTINEL = "===MODEL_DECK_HARVEST_PROBE==="

# Executed inside the running engine container. Prints the catalog once,
# bracketed by _SENTINEL lines. VLLM_LOGGING_STREAM is set before the vLLM
# import specifically (vLLM defaults it to stdout); the sentinel wrap is
# the general defense for chatter this module cannot enumerate in advance.
PROBE_SOURCE = '''
import json
import os

os.environ["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser

parser = make_arg_parser(FlexibleArgumentParser())
options = []
for action in parser._actions:
    if not action.option_strings:
        continue
    t = getattr(action, "type", None)
    # argparse's append actions are named "_AppendAction" /
    # "_AppendConstAction"; both, and any library subclass, contain
    # "Append" -- a repeatable flag takes the same option multiple times
    # to build a list, which the "text" fallback widget cannot express.
    options.append({
        "flags": action.option_strings,
        "type": getattr(t, "__name__", str(t) if t else None),
        "choices": list(action.choices) if action.choices else None,
        "default": repr(action.default),
        "nargs": action.nargs,
        "cls": type(action).__name__,
        "repeatable": "Append" in type(action).__name__,
        "help": action.help or "",
    })
print(SENTINEL)
print(json.dumps({"options": options}))
print(SENTINEL)
'''.replace("SENTINEL", repr(_SENTINEL))

_EXCLUDED_ACTIONS = ("_HelpAction", "_VersionAction")


def _extract_sentinel_payload(text: str) -> str:
    """Return the JSON between the LAST pair of _SENTINEL lines, or `text`
    unchanged when no sentinel pair is present -- keeps callers and
    fixtures that predate the sentinel (plain JSON, no wrapping) working
    unchanged. LAST pair, not first: chatter ahead of the real payload
    could itself coincidentally contain the sentinel text.
    """
    last = text.rfind(_SENTINEL)
    if last == -1:
        return text
    second_last = text.rfind(_SENTINEL, 0, last)
    if second_last == -1:
        return text
    return text[second_last + len(_SENTINEL):last]


def parse_probe_output(text: str, engine_version: str, now: str) -> dict:
    """Parse probe JSON into ONE characteristics field holding the catalog.

    Returns ``{}`` for unparseable output: a probe that failed must produce
    nothing rather than a half catalog that looks authoritative.
    """
    try:
        payload = json.loads(_extract_sentinel_payload(text))
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    options: dict[str, dict] = {}
    for entry in payload.get("options") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("cls") in _EXCLUDED_ACTIONS:
            continue
        flags = entry.get("flags") or []
        long_flags = [f for f in flags if f.startswith("--")]
        if not long_flags:
            continue
        canonical = long_flags[0]
        name = canonical.lstrip("-")
        options[name] = {
            # Every OTHER option string is an alias, not just the short
            # ones: BooleanOptionalAction pairs (--enable-x/--no-enable-x)
            # and mixed groups (--foo, --no-foo, -f) both carry a second
            # LONG flag that is just as real a spelling as the canonical.
            "aliases": [f for f in flags if f != canonical],
            "type": entry.get("type"),
            "choices": entry.get("choices"),
            "default": entry.get("default"),
            "nargs": entry.get("nargs"),
            "repeatable": entry.get("repeatable", False),
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
    if option.get("repeatable"):
        return "list"
    if option.get("choices"):
        return "select"
    if option.get("nargs") in ("+", "*"):
        return "list"
    if option.get("type") in ("int", "float"):
        return "number"
    return "text"
