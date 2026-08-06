"""Derive model facts from a local checkpoint directory.

Plain JSON reads only — no torch, no safetensors parsing, no weight
loading. A characteristics refresh runs on a timer and must stay cheap
enough that nobody is tempted to turn it off.

Absence is meaningful. A field that cannot be read is OMITTED, never set to
None: app.facts.detect_drift treats a missing fact as "cannot check", and a
None would turn every unquantized model into a spurious mismatch.

A corrupt or half-downloaded checkpoint degrades to fewer facts rather than
raising — one bad model directory must not take down the derive pass for
every other model.

Identity is the directory name, verbatim (the ontology's naming rule,
2026-08-04). Not an alias, not a normalization: the name on disk is the one
name, and 'aeon' is not a model.
"""

import json
from pathlib import Path

_CONFIG = "config.json"
_GENERATION_CONFIG = "generation_config.json"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_CHAT_TEMPLATE = "chat_template.jinja"

_SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def derive_checkpoint(path: Path, now: str) -> dict[str, dict]:
    """Return ``{field: {value, source, derived_ts}}`` for the checkpoint at
    `path`. An unreadable directory yields ``{}``.
    """
    if not path.is_dir():
        return {}

    facts: dict[str, dict] = {
        "identity": _field(path.name, "directory name", now),
    }

    config = _read_json(path / _CONFIG)

    quant = (config.get("quantization_config") or {}).get("quant_method")
    if quant is not None:
        facts["quant_method"] = _field(quant, _CONFIG, now)

    max_pos = config.get("max_position_embeddings")
    if max_pos is not None:
        facts["max_position_embeddings"] = _field(max_pos, _CONFIG, now)

    architectures = config.get("architectures") or []
    if architectures:
        facts["architecture"] = _field(architectures[0], _CONFIG, now)

    generation = _read_json(path / _GENERATION_CONFIG)
    sampling = {k: generation[k] for k in _SAMPLING_KEYS if k in generation}
    if sampling:
        facts["recommended_sampling"] = _field(sampling, _GENERATION_CONFIG, now)

    template_source = _chat_template_source(path)
    if template_source is not None:
        facts["chat_template_present"] = _field(True, template_source, now)

    return facts


def _chat_template_source(path: Path) -> str | None:
    if (path / _CHAT_TEMPLATE).is_file():
        return _CHAT_TEMPLATE
    tokenizer = _read_json(path / _TOKENIZER_CONFIG)
    if tokenizer.get("chat_template"):
        return _TOKENIZER_CONFIG
    return None


def _read_json(path: Path) -> dict:
    """Read a JSON object, or {} for missing/corrupt/non-object content."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _field(value, source: str, now: str) -> dict:
    return {"value": value, "source": source, "derived_ts": now}
