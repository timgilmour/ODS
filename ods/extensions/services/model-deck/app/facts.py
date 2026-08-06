"""Model Deck fact resolution and drift detection — pure functions.

``resolve_facts`` merges the derived cache with the declared allowlist,
declared winning, and keeps the shadowed derived value so the UI can show
"a human asserted this, overriding what we read". Origin is never dropped:
an operator has to be able to see that ``tools_verified`` is an assertion
someone made, not something measured.

``detect_drift`` is the payoff of the whole layer. It compares facts that
should agree and reports when they don't — it never corrects them. Every
rule here comes from a real incident:

* **quantization** (severity ``crash``): a profile forcing
  ``--quantization modelopt`` on a checkpoint whose config.json says
  ``compressed-tensors`` will not boot. heretic and ornith carried exactly
  that from their creation until 2026-07-31, crash-looping every swap. Note
  the inverse is legitimate — mm27b really is modelopt — so the rule
  compares values rather than blacklisting one.
* **max_model_len** (severity ``mismatch``): serving above the checkpoint's
  ``max_position_embeddings`` is a misconfiguration.
* **max_input_tokens** (severity ``mismatch``): what the gateway advertises
  versus what the engine actually serves. Context currently lives in three
  unreconciled places; this is the check that notices.

Absence is never drift. A remote model with no readable checkpoint yields
no facts, and "cannot check" must never render as "mismatch" — false alarms
are how real alarms get ignored.
"""

# runtime_field -> (fact_field, comparison, severity)
#   "eq"  : drift when values differ
#   "lte" : drift when the runtime value EXCEEDS the fact
DRIFT_RULES = {
    "quantization": ("quant_method", "eq", "crash"),
    "max_model_len": ("max_position_embeddings", "lte", "mismatch"),
    "max_input_tokens": ("max_model_len_live", "eq", "mismatch"),
}


def resolve_facts(derived: dict, declared: dict) -> dict:
    """Merge derived and declared facts, declared winning, origin retained."""
    resolved: dict[str, dict] = {}

    for name, field in derived.items():
        resolved[name] = {
            "value": field["value"],
            "origin": "derived",
            "source": field["source"],
            "derived_ts": field["derived_ts"],
        }

    for name, value in declared.items():
        entry = {
            "value": value,
            "origin": "declared",
            "source": "declared.json",
            "derived_ts": None,
        }
        if name in resolved:
            entry["shadowed_value"] = resolved[name]["value"]
        resolved[name] = entry

    return resolved


def detect_drift(facts: dict, runtime: dict) -> list[dict]:
    """Report where `runtime` config contradicts resolved `facts`.

    `runtime` is the flat config actually in force for a placement
    (profile flags, gateway metadata). Returns one entry per disagreement,
    in DRIFT_RULES order. Never mutates, never corrects.
    """
    drift = []

    for runtime_field, (fact_field, comparison, severity) in DRIFT_RULES.items():
        if runtime_field not in runtime:
            continue
        fact = facts.get(fact_field)
        if fact is None:
            # Nothing authoritative to compare against — cannot check is
            # not the same as mismatch.
            continue

        actual = runtime[runtime_field]
        expected = fact["value"]

        if comparison == "eq" and actual == expected:
            continue
        if comparison == "lte" and _lte(actual, expected):
            continue

        drift.append({
            "field": runtime_field,
            "expected": expected,
            "actual": actual,
            "expected_source": fact["source"],
            "actual_source": "runtime config",
            "severity": severity,
        })

    return drift


def _lte(actual, expected) -> bool:
    """True when `actual` is within `expected`. Non-numeric values cannot be
    compared this way and are treated as within-bounds rather than raising:
    a malformed config should surface as its own problem, not as a crash in
    drift detection."""
    try:
        return float(actual) <= float(expected)
    except (TypeError, ValueError):
        return True
