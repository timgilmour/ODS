"""Tests for app.facts — merge with provenance, and drift detection.

detect_drift is the payoff of the whole characteristics layer: it is the
check that would have caught --quantization modelopt against a
compressed-tensors checkpoint BEFORE the crash loop, instead of after three
profiles had been broken for weeks.
"""

from app.facts import detect_drift, resolve_facts


def _derived(**fields):
    return {name: {"value": v, "source": "config.json", "derived_ts": "t"}
            for name, v in fields.items()}


def test_derived_field_carries_origin_derived():
    resolved = resolve_facts(_derived(quant_method="compressed-tensors"), {})

    assert resolved["quant_method"]["origin"] == "derived"
    assert resolved["quant_method"]["value"] == "compressed-tensors"
    assert resolved["quant_method"]["source"] == "config.json"


def test_declared_field_carries_origin_declared():
    resolved = resolve_facts({}, {"tools_verified": True})

    assert resolved["tools_verified"] == {
        "value": True, "origin": "declared", "source": "declared.json", "derived_ts": None,
    }


def test_declared_wins_over_derived_for_the_same_field():
    """Declared-over-derived, but the UI must still be able to say the
    derived value existed — so it is kept as shadowed_value."""
    resolved = resolve_facts(_derived(label="auto"), {"label": "human"})

    assert resolved["label"]["value"] == "human"
    assert resolved["label"]["origin"] == "declared"
    assert resolved["label"]["shadowed_value"] == "auto"


def test_fields_are_unioned_not_intersected():
    resolved = resolve_facts(_derived(a=1), {"label": "x"})

    assert set(resolved) == {"a", "label"}


def test_empty_inputs_resolve_to_empty():
    assert resolve_facts({}, {}) == {}


def test_quantization_conflict_is_severity_crash():
    """THE regression: forcing modelopt on a compressed-tensors checkpoint
    crash-looped heretic and ornith from 2026-07-31."""
    facts = resolve_facts(_derived(quant_method="compressed-tensors"), {})

    drift = detect_drift(facts, {"quantization": "modelopt"})

    assert len(drift) == 1
    assert drift[0]["field"] == "quantization"
    assert drift[0]["expected"] == "compressed-tensors"
    assert drift[0]["actual"] == "modelopt"
    assert drift[0]["severity"] == "crash"


def test_quantization_agreement_is_not_drift():
    """mm27b IS genuinely modelopt. A rule that flags it would be a false
    positive that teaches everyone to ignore the warnings."""
    facts = resolve_facts(_derived(quant_method="modelopt"), {})

    assert detect_drift(facts, {"quantization": "modelopt"}) == []


def test_absent_quantization_flag_is_not_drift():
    """No flag means 'let vLLM auto-detect', which is the correct config
    for a compressed-tensors checkpoint (heretic's fix)."""
    facts = resolve_facts(_derived(quant_method="compressed-tensors"), {})

    assert detect_drift(facts, {}) == []


def test_context_above_checkpoint_capability_is_mismatch():
    facts = resolve_facts(_derived(max_position_embeddings=262144), {})

    drift = detect_drift(facts, {"max_model_len": 999999})

    assert drift[0]["field"] == "max_model_len"
    assert drift[0]["severity"] == "mismatch"


def test_context_at_or_below_capability_is_not_drift():
    facts = resolve_facts(_derived(max_position_embeddings=262144), {})

    assert detect_drift(facts, {"max_model_len": 131072}) == []
    assert detect_drift(facts, {"max_model_len": 262144}) == []


def test_gateway_context_disagreeing_with_live_is_mismatch():
    """max_input_tokens in extra-routes.json vs what the engine actually
    serves — the third place context is written down."""
    facts = resolve_facts(_derived(max_model_len_live=262144), {})

    drift = detect_drift(facts, {"max_input_tokens": 131072})

    assert drift[0]["field"] == "max_input_tokens"


def test_missing_derived_fact_yields_no_drift_not_a_crash():
    """Remote models have no readable checkpoint in this plan. Absence of a
    fact must mean 'cannot check', never 'mismatch'."""
    assert detect_drift({}, {"quantization": "modelopt"}) == []


def test_drift_entries_name_both_sources():
    facts = resolve_facts(_derived(quant_method="compressed-tensors"), {})

    drift = detect_drift(facts, {"quantization": "modelopt"})

    assert drift[0]["expected_source"] == "config.json"
    assert drift[0]["actual_source"]
