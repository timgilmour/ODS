#!/usr/bin/env python3
"""Stage (write/remove) ONE instance's entry in the ODS gateway's litellm
extra-routes sidecar (INST I1, D-I1-4). This is the renderer's INPUT
(`ods/scripts/render-runtime-configs.py`'s `load_extra_litellm_routes` reads
`<ods-dir>/config/litellm/extra-routes.json`); APPLYING a staged route —
regenerating the litellm config and recreating the container — is the
existing ODS render+litellm-recreate path (activate/regen) and is NOT
triggered here. Only kinds that serve a model at boot (their template's
"route" is non-null) get an entry; exit 0 no-op otherwise.

Exit 0: staged (or a legitimate no-op). Exit 1: the sidecar exists and is
malformed — refused, never overwritten (silently dropping operator routes
would recreate the exact failure the sidecar exists to fix; same posture
as the loader it feeds).
"""
import json
import os
import sys

# Entries owned by this instance carry this marker; the ODS loader
# (render-runtime-configs.py's load_extra_litellm_routes) copies only its
# named fields, so the marker never reaches the rendered litellm config.
OWNER_KEY = "_deck_instance"
# The host a staged route dials: the instance's compose service name == its
# container name (render_instance.CONTAINER_PREFIX; pinned to the deck's
# INSTANCE_CONTAINER_PREFIX by model-deck/tests/test_instances_parity.py).
CONTAINER_PREFIX = "deck-"

# Reserved model_name values, copied from
# ods/scripts/render-runtime-configs.py:324-327 (_EXTRA_ROUTES_RESERVED) —
# an instance must never shadow a core route.
_EXTRA_ROUTES_RESERVED = {
    "default", "*", "hipfire", "lemonade", "local", "cloud", "ods/current",
}


def _die(code: int, msg: str) -> None:
    sys.stderr.write(f"stage_route: {msg}\n")
    sys.exit(code)


def _load_sidecar(path: str):
    """Return the sidecar's entry list, or None if it does not exist yet.
    Exits 1 if it exists but is not a well-formed JSON list of objects."""
    if not os.path.isfile(path):
        return None
    try:
        data = json.loads(open(path).read())
    except ValueError as exc:
        _die(1, f"{path} is not valid JSON, refusing to touch it: {exc}")
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        _die(1, f"{path} must be a JSON list of route objects, refusing to touch it")
    return data


def _write_sidecar(path: str, entries: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(entries, fh, indent=2)
    os.replace(tmp, path)


def main(verb, doc_path, templates_dir, sidecar_path) -> None:
    doc = json.load(open(doc_path))
    resource, kind = doc["resource"], doc["kind"]

    kinds = json.load(open(os.path.join(templates_dir, "kinds.json")))
    tpl = json.load(open(os.path.join(templates_dir, kinds[kind])))
    route = tpl.get("route")

    entries = _load_sidecar(sidecar_path)  # exits 1 itself if malformed

    if verb == "remove":
        if entries is None:
            return  # nothing staged for this resource, nothing to do
        kept = [e for e in entries if e.get(OWNER_KEY) != resource]
        if kept != entries:
            _write_sidecar(sidecar_path, kept)
        return

    # create / move
    if route is None:
        return  # this kind serves no model at boot (lemonade/comfyui — later increment)

    model_env = route.get("model_env")
    env = doc.get("env", {})
    model = env.get(model_env) if model_env else None
    if not model:
        sys.stderr.write(
            f"stage_route: document env has no {model_env!r} for kind {kind!r}; no-op\n")
        return

    entries = entries or []
    others = [e for e in entries if e.get(OWNER_KEY) != resource]
    taken = {e.get("model_name") for e in others}
    model_name = model
    if model_name in _EXTRA_ROUTES_RESERVED or model_name in taken:
        model_name = f"{model}-{resource}"

    internal_port = tpl["internal_port"]
    path = route.get("path", "")
    new_entry = {
        "model_name": model_name,
        "model": f"openai/{model}",
        "api_base": f"http://{CONTAINER_PREFIX}{resource}:{internal_port}{path}",
        OWNER_KEY: resource,
    }
    _write_sidecar(sidecar_path, others + [new_entry])


if __name__ == "__main__":
    if len(sys.argv) != 5:
        _die(1, "usage: stage_route.py <verb> <document.json> <templates-dir> <extra-routes.json>")
    main(*sys.argv[1:5])
