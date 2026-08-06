"""Derive facts from an engine's live OpenAI-style /v1/models surface.

This is the only authoritative source for a REMOTE model's facts in this
increment: with no node-agent changes, remote checkpoint files are
unreadable. A single-slot node answers for exactly one model at a time, so
what comes back here is inherently partial — everything else about the
other models on that node stays declared, and must be labelled as such.

Facts derived here are true only while the model is loaded. They are stamped
with source "/v1/models" so a reader can tell them from checkpoint facts,
which remain true whether or not anything is running.

Absence is meaningful (same rule as app.derive_checkpoint): a field the
engine does not report is omitted, never None.
"""

_SOURCE = "/v1/models"


def derive_live_models(models_body: dict, now: str) -> dict[str, dict[str, dict]]:
    """Map a /v1/models response to ``{model_id: {field: record}}``."""
    facts: dict[str, dict[str, dict]] = {}

    for entry in models_body.get("data") or []:
        model_id = entry.get("id")
        if not model_id:
            continue

        fields: dict[str, dict] = {
            "served": _field(True, now),
        }
        if entry.get("max_model_len") is not None:
            fields["max_model_len_live"] = _field(entry["max_model_len"], now)

        facts[model_id] = fields

    return facts


def _field(value, now: str) -> dict:
    return {"value": value, "source": _SOURCE, "derived_ts": now}
