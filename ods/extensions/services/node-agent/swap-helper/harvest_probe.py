
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
print('===MODEL_DECK_HARVEST_PROBE===')
print(json.dumps({"options": options}))
print('===MODEL_DECK_HARVEST_PROBE===')
