"""stage_route.py contract tests.

INST I1 D-I1-4: the instances-helper stages (writes/removes) the instance's
entry in the ODS gateway's litellm extra-routes sidecar
(`ods/config/litellm/extra-routes.json`, the INPUT to
`render-runtime-configs.py`'s `load_extra_litellm_routes`) after a successful
create/remove/move. Staging never APPLIES the route (that is the existing
ODS render + litellm recreate path, not triggered here) and never fails the
verb — the helper logs a staging failure but always returns ok for the verb
itself (covered by test_instances_helper.py, not here).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[1] / "instances-helper" / "stage_route.py"
TEMPLATES = Path(__file__).resolve().parents[1] / "instances-helper" / "templates"
DOC = {"resource": "hipfire-2", "kind": "hipfire", "gpu_indices": [1], "port": 11501,
       "env": {"HIPFIRE_MODEL": "qwen3.8:27b"}}


def _stage(tmp_path, verb, doc, sidecar_content=None):
    docp = tmp_path / "doc.json"
    docp.write_text(json.dumps(doc))
    side = tmp_path / "extra-routes.json"
    if sidecar_content is not None:
        side.write_text(sidecar_content)
    r = subprocess.run([sys.executable, str(STAGE), verb, str(docp), str(TEMPLATES), str(side)],
                        capture_output=True, text=True)
    return r, side


def test_create_appends_an_owned_entry_named_by_the_boot_model(tmp_path):
    r, side = _stage(tmp_path, "create", DOC)
    assert r.returncode == 0, r.stderr
    assert json.loads(side.read_text()) == [{"model_name": "qwen3.8:27b", "model": "openai/qwen3.8:27b",
                                             "api_base": "http://hipfire-2:11435/v1", "_deck_instance": "hipfire-2"}]


def test_taken_or_reserved_name_gets_the_instance_suffix(tmp_path):
    existing = json.dumps([{"model_name": "qwen3.8:27b", "model": "openai/x", "api_base": "http://other:1/v1"}])
    _, side = _stage(tmp_path, "create", DOC, existing)
    names = [e["model_name"] for e in json.loads(side.read_text())]
    assert names == ["qwen3.8:27b", "qwen3.8:27b-hipfire-2"]
    _, side2 = _stage(tmp_path, "create", {**DOC, "env": {"HIPFIRE_MODEL": "hipfire"}})   # reserved core name
    # NOTE: brief text says "[0]"; that contradicts the brief's own documented
    # algorithm ("remove the resource's own old entries first then append")
    # together with the equality check two lines up (which requires the new
    # entry to land at the TAIL, not the head, when an unrelated entry is
    # already present) — [-1] is the newly-staged entry under append semantics.
    assert json.loads(side2.read_text())[-1]["model_name"] == "hipfire-hipfire-2"


def test_remove_drops_only_the_owned_entries(tmp_path):
    existing = json.dumps([{"model_name": "spark-x", "model": "openai/x", "api_base": "http://s:1/v1"},
                           {"model_name": "m", "model": "openai/m", "api_base": "http://hipfire-2:11435/v1", "_deck_instance": "hipfire-2"}])
    _, side = _stage(tmp_path, "remove", DOC, existing)
    assert [e["model_name"] for e in json.loads(side.read_text())] == ["spark-x"]


def test_kind_without_a_route_block_is_a_no_op(tmp_path):
    r, side = _stage(tmp_path, "create", {**DOC, "kind": "lemonade", "env": {}})
    assert r.returncode == 0 and not side.exists()


def test_malformed_sidecar_is_refused_not_overwritten(tmp_path):
    r, side = _stage(tmp_path, "create", DOC, "{not json")
    assert r.returncode == 1 and side.read_text() == "{not json"


def test_ods_loader_ignores_the_ownership_marker(tmp_path):
    # pins D-I1-4's assumption against the real loader
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location("rrc", Path(__file__).resolve().parents[4] / "scripts" / "render-runtime-configs.py")
    if spec is None or not Path(spec.origin).exists():
        pytest.skip("needs the full repo checkout")
    rrc = importlib.util.module_from_spec(spec)
    _sys.modules["rrc"] = rrc  # dataclasses in this module resolve string annotations via sys.modules[cls.__module__]
    spec.loader.exec_module(rrc)
    _, side = _stage(tmp_path, "create", DOC)
    (routes,) = rrc.load_extra_litellm_routes(str(side))
    assert routes["model_name"] == "qwen3.8:27b" and "_deck_instance" not in routes
