"""Tests for app.derive_live — an engine's /v1/models -> derived facts.

This is the ONLY authoritative source for remote models in this plan (no
node-agent changes, so remote checkpoint files are unreadable). A single-
slot node answers for exactly one model at a time, so these facts are
inherently partial and must say so.
"""

from app.derive_live import derive_live_models


def test_served_context_derived_from_max_model_len():
    body = {"data": [{"id": "Qwen3.6-35B-A3B-heretic-NVFP4", "max_model_len": 262144}]}

    facts = derive_live_models(body, now="t")

    entry = facts["Qwen3.6-35B-A3B-heretic-NVFP4"]
    assert entry["max_model_len_live"]["value"] == 262144
    assert entry["max_model_len_live"]["source"] == "/v1/models"


def test_multiple_models_each_get_an_entry():
    body = {"data": [{"id": "a", "max_model_len": 1}, {"id": "b", "max_model_len": 2}]}

    facts = derive_live_models(body, now="t")

    assert set(facts) == {"a", "b"}


def test_entry_without_max_model_len_yields_no_context_field():
    """Absent, not None — detect_drift relies on absence meaning
    'cannot check'."""
    body = {"data": [{"id": "a"}]}

    assert "max_model_len_live" not in derive_live_models(body, now="t")["a"]


def test_model_still_gets_an_entry_recording_that_it_is_loaded():
    body = {"data": [{"id": "a"}]}

    facts = derive_live_models(body, now="t")

    assert facts["a"]["served"]["value"] is True


def test_empty_body_yields_nothing():
    assert derive_live_models({}, now="t") == {}
    assert derive_live_models({"data": []}, now="t") == {}


def test_entry_without_id_is_skipped_not_crashed():
    body = {"data": [{"max_model_len": 1}, {"id": "ok"}]}

    assert set(derive_live_models(body, now="t")) == {"ok"}


def test_every_field_stamped_with_now():
    body = {"data": [{"id": "a", "max_model_len": 1}]}

    facts = derive_live_models(body, now="2026-08-04T00:00:00+00:00")

    assert all(f["derived_ts"] == "2026-08-04T00:00:00+00:00" for f in facts["a"].values())
