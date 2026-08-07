"""Settings validation — three warning classes, and never a refusal.

Nothing here blocks a save. That is a deliberate choice with a cost
asymmetry behind it: blocking an unrecognised flag means a person cannot
reach a knob their engine genuinely has (new upstream version, stale
catalog), while warning about a wrong one costs a yellow chip. The escape
hatch has to exist, so it exists by default.

Three classes:

* ``unknown``  — not in the harvested catalog (or there is no catalog),
  and not a known alias of something that is.
* ``type``     — contradicts the harvested schema (type, choices, arity).
* ``conflict`` — contradicts a DERIVED fact about the model. This is the
  crash-loop catcher: ``--quantization modelopt`` against a checkpoint whose
  config.json says ``compressed-tensors`` will not boot, and heretic and
  ornith carried exactly that from creation until 2026-07-31.

Severity vocabulary actually emitted (cite this, not the mockup): every
``unknown`` and ``type`` warning below carries ``severity: "warn"``, set
right here. ``conflict`` warnings carry whatever ``app.facts.DRIFT_RULES``
assigned that rule — today either ``"crash"`` or ``"mismatch"`` — passed
through unmodified from ``app.facts.detect_drift``. This module never
invents a ``conflict`` severity of its own; a UI comparing against a fourth
value (e.g. a mockup's vocabulary) will never see it fire. See
``app/facts.py`` for the rule table.

Note the conflict rule compares values rather than blacklisting one:
mm27b genuinely IS modelopt, and a false positive there would teach
everyone to ignore the warnings — which is how a warning system dies.

Absence never produces a conflict. A remote model has no readable
checkpoint; "cannot check" must not render as "mismatch".

Catalog aliases: the harvested catalog's ``options[*].aliases`` carries
every OTHER spelling of an option — short forms (``-q``) and, for
``BooleanOptionalAction`` pairs, the ``--no-*`` half, which is the ONLY
spelling of the negative sense (there is no positive-form way to write
"disable this"). A settings key is matched against the catalog by trying
the canonical name first and falling back to an alias index built from
every option's ``aliases`` list, dash-stripped the same way
``app.argline`` strips settings keys. The canonical option entry found
this way is what type/choices/arity validation runs against — the warning
itself still names the ORIGINAL settings key, never the canonical one, so
a caller can always find its own key in the results.

List-valued settings: ``list[str]`` is a legitimate resolved value for a
repeatable flag (argparse "Append" actions) or an ``nargs in ("+", "*")``
option — not a shape error. A list is validated element-wise against the
catalog's ``type``/``choices``; whether a list is legal AT ALL for a given
option is decided by that option's ``repeatable``/``nargs`` fields, so a
list handed to a single-value option still warns.

Pure functions, no I/O. Reuses app.facts.detect_drift for the conflict
class so there is exactly one definition of what disagreement means.
``detect_drift`` speaks the runtime-config vocabulary (underscored field
names: ``max_model_len``, ``max_input_tokens``, ``quantization``); every
settings key is dash-form. The bridge between the two is built
mechanically from whatever keys are actually present in ``resolved`` (not
hand-listed per rule), and a reverse map turns detect_drift's echoed field
name back into the original dash-form settings key before it becomes a
warning — so every conflict warning's ``key`` is something a settings-key
consumer can actually find.

Never raises. A malformed ``resolved`` entry (not a dict, or a dict with
no ``"value"``) is skipped rather than crashing the whole validation pass
— warn-not-crash applies to validate_settings's own input handling, not
just to the values it reports on.
"""

from app.argline import POSITIONAL_KEY
from app.facts import detect_drift

_NUMERIC_TYPES = ("int", "float")
_LIST_NARGS = ("+", "*")


def validate_settings(resolved: dict, catalog: dict | None, facts: dict) -> list[dict]:
    """Return warnings for `resolved` settings. Never raises, never blocks."""
    issues: list[dict] = []
    options = (catalog or {}).get("options", {})
    alias_index = _alias_index(options)

    for key, entry in resolved.items():
        if key == POSITIONAL_KEY:
            continue
        if not isinstance(entry, dict) or "value" not in entry:
            # Malformed shape leaking past the resolved-value invariant
            # (RULING 2026-08-07, Task 3 review amendment): warn-not-crash
            # means this key just isn't checked, not that validation raises.
            continue

        value = entry["value"]
        canonical = key if key in options else alias_index.get(key)
        option = options.get(canonical) if canonical is not None else None

        if option is None:
            issues.append({
                "key": key,
                "class": "unknown",
                "severity": "warn",
                "message": (
                    f"{key!r} is not in this engine's option catalog — it will still be "
                    "applied, but nothing has validated it"
                ),
            })
            continue

        message = _type_problem(option, value)
        if message:
            issues.append({
                "key": key, "class": "type", "severity": "warn",
                "message": f"{key}: {message}",
            })

    issues.extend(_conflicts(resolved, facts))
    return issues


def _alias_index(options: dict) -> dict[str, str]:
    """Map every alias spelling (dash-stripped, same normalization
    ``app.argline`` applies to settings keys) to its canonical catalog
    name. Without this, ``no-enable-prefix-caching`` (the only spelling of
    that BooleanOptionalAction's negative half) or a short form like ``q``
    has no catalog entry to match against and falsely reads as unknown."""
    index: dict[str, str] = {}
    for name, option in options.items():
        for alias in option.get("aliases") or []:
            index[alias.lstrip("-")] = name
    return index


def _type_problem(option: dict, value) -> str | None:
    if isinstance(value, list):
        return _list_type_problem(option, value)
    return _scalar_type_problem(option, value)


def _list_type_problem(option: dict, value: list) -> str | None:
    """A list is only legal on a repeatable flag or an nargs +/* option —
    consult the catalog rather than assume. When it is legal, validate
    each element against the same rules a scalar would face."""
    if not (option.get("repeatable") or option.get("nargs") in _LIST_NARGS):
        return f"{value!r} is a list but this option takes a single value"

    problems = [p for p in (_scalar_type_problem(option, v) for v in value) if p]
    return "; ".join(problems) if problems else None


def _scalar_type_problem(option: dict, value) -> str | None:
    if option.get("nargs") == 0 and value is not True:
        return f"{value!r} was given but this is a flag — it takes no value"

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
    """Conflicts with derived facts, via the single shared drift definition.

    Builds the dash->underscore bridge mechanically from every well-formed
    key in `resolved` (not a per-rule hand list), so any current or future
    DRIFT_RULES field is reachable from settings without a special case —
    and keeps a reverse map so the emitted warning's `key` is always the
    original dash-form settings key, never detect_drift's echoed field
    name.
    """
    runtime: dict[str, object] = {}
    original_key: dict[str, str] = {}

    for key, entry in resolved.items():
        if key == POSITIONAL_KEY or not isinstance(entry, dict) or "value" not in entry:
            continue
        bridged_key = key.replace("-", "_")
        runtime[bridged_key] = entry["value"]
        original_key[bridged_key] = key

    return [
        {
            "key": original_key.get(entry["field"], entry["field"]),
            "class": "conflict",
            "severity": entry["severity"],
            "message": (
                f"{entry['field']} is {entry['actual']!r} but {entry['expected_source']} "
                f"says {entry['expected']!r}"
            ),
        }
        for entry in detect_drift(facts, runtime)
    ]
