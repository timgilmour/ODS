#!/usr/bin/env python3
"""Render ONE instance document into ONE compose file (JSON, which is valid
YAML) for project `deck-instances`. Exit 0 written / 1 malformed document or
env outside the kind's allowlist / 2 unknown kind. Nothing is ever launched
from a document this script refused — the helper only runs `docker compose`
on a file this script wrote. Kind → template is DATA (templates/kinds.json),
the single seam E2 can turn into a served descriptor.

Instances own the `deck-*` DNS namespace on ods-network: the compose SERVICE
name (the alias every deck/litellm URL dials) IS the container name,
`deck-<resource>`. That is what keeps an instance from ever shadowing an ODS
service alias (hipfire, qdrant, dashboard-api, …) — a property of the naming
scheme, not of a reserved-name list that would rot as the stack grows."""
import json
import os
import re
import sys

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*\Z")
TEMPLATE_KEYS = {"image", "internal_port", "service", "environment", "env_allow",
                 "volumes", "per_instance_dirs", "route"}
# Service name == container name == deck-<resource>. Pinned equal to the
# deck's INSTANCE_CONTAINER_PREFIX by model-deck/tests/test_instances_parity.py.
CONTAINER_PREFIX = "deck-"
# Keys the renderer itself sets on the service block (container_name/image
# from the template + doc; environment/ports/volumes/networks computed from
# the doc and template's own top-level fields). A template that also sets
# one of these in its "service" block would have it silently overwritten
# (or would silently win, if merge order ever changed) — refuse instead.
RENDERER_OWNED_SERVICE_KEYS = {"container_name", "image", "environment", "ports",
                               "volumes", "networks"}


def _die(code: int, msg: str) -> None:
    sys.stderr.write(f"render_instance: {msg}\n")
    sys.exit(code)


def main(templates_dir, doc_path, out_path, instances_dir, ods_dir) -> None:
    try:
        doc = json.load(open(doc_path))
        resource, kind = doc["resource"], doc["kind"]
        gpus, port, env = doc["gpu_indices"], doc["port"], doc["env"]
        assert isinstance(resource, str) and NAME_RE.match(resource), "resource"
        assert isinstance(gpus, list) and gpus and all(isinstance(g, int) and not isinstance(g, bool) and g >= 0 for g in gpus), "gpu_indices"
        assert isinstance(port, int) and not isinstance(port, bool) and 1024 <= port <= 65535, "port"
        assert isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), "env"
    except Exception as exc:  # noqa: BLE001 — every malformation is the same answer: refuse, say so
        _die(1, f"unusable instance document {doc_path}: {exc!r}")
    kinds = json.load(open(os.path.join(templates_dir, "kinds.json")))
    if kind not in kinds:
        _die(2, f"unknown kind {kind!r} (known: {sorted(kinds)})")
    tpl_path = os.path.join(templates_dir, kinds[kind])
    if not os.path.isfile(tpl_path):
        _die(2, f"unknown kind {kind!r} (template file missing: {kinds[kind]})")
    tpl = json.load(open(tpl_path))
    if set(tpl) != TEMPLATE_KEYS:
        _die(1, f"template {kinds[kind]} must have exactly {sorted(TEMPLATE_KEYS)}")
    owned = sorted(set(tpl["service"]) & RENDERER_OWNED_SERVICE_KEYS)
    if owned:
        _die(1, f"template {kinds[kind]} service block sets renderer-owned key(s) {owned}")
    bad = sorted(set(env) - set(tpl["env_allow"]))
    if bad:
        _die(1, f"env {bad} not allowed for kind {kind!r} (allowed: {tpl['env_allow']})")
    data_dir = os.path.join(instances_dir, "data", resource)
    for sub in tpl["per_instance_dirs"]:
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)
    container = f"{CONTAINER_PREFIX}{resource}"
    subst = {"resource": resource, "container": container, "data_dir": data_dir,
             "ods_dir": ods_dir, "port": str(port)}
    service = {
        "container_name": container,
        "image": tpl["image"],
        **tpl["service"],
        "environment": {**tpl["environment"], **env,
                        # ALWAYS explicit — a bare ROCR_VISIBLE_DEVICES in the
                        # operator's shell/.env must never leak into an instance.
                        "ROCR_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)},
        "ports": [f"127.0.0.1:{port}:{tpl['internal_port']}"],
        "volumes": [v.format_map(subst) for v in tpl["volumes"]],
        "networks": ["ods"],
    }
    out = {"services": {container: service},
           "networks": {"ods": {"external": True, "name": "ods-network"}}}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        _die(1, "usage: render_instance.py <templates-dir> <document.json> <out.yaml> <instances-dir> <ods-dir>")
    main(*sys.argv[1:6])
