"""Settings validation — three warning classes, and never a refusal.

Nothing here blocks a save. That is a deliberate choice with a cost
asymmetry behind it: blocking an unrecognised flag means a person cannot
reach a knob their engine genuinely has (new upstream version, stale
catalog), while warning about a wrong one costs a yellow chip. The escape
hatch has to exist, so it exists by default.

Three classes:

* ``unknown``  — not in the harvested catalog (or there is no catalog).
* ``type``     — contradicts the harvested schema (type or choices).
* ``conflict`` — contradicts a DERIVED fact about the model. This is the
  crash-loop catcher: ``--quantization modelopt`` against a checkpoint whose
  config.json says ``compressed-tensors`` will not boot, and heretic and
  ornith carried exactly that from creation until 2026-07-31.

Note the conflict rule compares values rather than blacklisting one:
mm27b genuinely IS modelopt, and a false positive there would teach
everyone to ignore the warnings — which is how a warning system dies.

Absence never produces a conflict. A remote model has no readable
checkpoint; "cannot check" must not render as "mismatch".

Pure functions, no I/O. Reuses app.facts.detect_drift for the conflict
class so there is exactly one definition of what disagreement means.
"""

from app.argline import POSITIONAL_KEY
from app.facts import detect_drift

_NUMERIC_TYPES = ("int", "float")


def validate_settings(resolved: dict, catalog: dict | None, facts: dict) -> list[dict]:
    """Return warnings for `resolved` settings. Never raises, never blocks."""
    warnings: list[dict] = []
    options = (catalog or {}).get("options", {})

    for key, entry in resolved.items():
        if key == POSITIONAL_KEY:
            continue

        option = options.get(key)
        if option is None:
            warnings.append({
                "key": key,
                "class": "unknown",
                "severity": "warn",
                "message": (
                    f"{key!r} is not in this engine's option catalog — it will still be "
                    "applied, but nothing has validated it"
                ),
            })
            continue

        message = _type_problem(option, entry["value"])
        if message:
            warnings.append({
                "key": key, "class": "type", "severity": "warn",
                "message": f"{key}: {message}",
            })

    warnings.extend(_conflicts(resolved, facts))
    return warnings


def _type_problem(option: dict, value) -> str | None:
    choices = option.get("choices")
    if choices and value not in choices:
        return f"{value!r} is not one of {choices}"

    if option.get("type") in _NUMERIC_TYPES and value is not True:
        try:
            float(value)
        except (TypeError, ValueError):
            return f"{value!r} is not a {option['type']}"

    return None


def _conflicts(resolved: dict, facts: dict) -> list[dict]:
    """Conflicts with derived facts, via the single shared drift definition."""
    runtime = {key: entry["value"] for key, entry in resolved.items() if key != POSITIONAL_KEY}
    # detect_drift speaks the runtime-config vocabulary; settings keys are
    # dashed flags, so bridge the two names it cares about.
    bridged = dict(runtime)
    if "max-model-len" in runtime:
        bridged["max_model_len"] = runtime["max-model-len"]

    return [
        {
            "key": entry["field"],
            "class": "conflict",
            "severity": entry["severity"],
            "message": (
                f"{entry['field']} is {entry['actual']!r} but {entry['expected_source']} "
                f"says {entry['expected']!r}"
            ),
        }
        for entry in detect_drift(facts, bridged)
    ]
