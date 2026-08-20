import json, subprocess, sys
from pathlib import Path
import pytest

RENDER = Path(__file__).resolve().parents[1] / "instances-helper" / "render_instance.py"
TEMPLATES = Path(__file__).resolve().parents[1] / "instances-helper" / "templates"
DOC = {"resource": "agent", "kind": "hipfire", "gpu_indices": [2], "port": 11500,
       "env": {"HIPFIRE_MODEL": "qwen3.8:27b"}}


def _render(tmp_path, doc, templates=TEMPLATES):
    docp = tmp_path / "doc.json"; docp.write_text(json.dumps(doc))
    out = tmp_path / "out.yaml"
    inst = tmp_path / "instances"; inst.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(RENDER), str(templates), str(docp), str(out),
                        str(inst), "/opt/ods"], capture_output=True, text=True)
    return r, out, inst


def test_hipfire_renders_service_named_by_resource_with_explicit_rocr_and_host_port(tmp_path):
    r, out, inst = _render(tmp_path, DOC)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())             # JSON is valid YAML; the renderer emits JSON
    svc = doc["services"]["agent"]
    assert svc["container_name"] == "deck-agent"
    assert svc["environment"]["ROCR_VISIBLE_DEVICES"] == "2"
    assert svc["environment"]["HIPFIRE_MODEL"] == "qwen3.8:27b"
    assert "HSA_OVERRIDE_GFX_VERSION" not in svc["environment"]
    assert svc["ports"] == ["127.0.0.1:11500:11435"]
    assert svc["networks"] == ["ods"]
    assert doc["networks"] == {"ods": {"external": True, "name": "ods-network"}}
    assert f"{inst}/data/agent/kernels:/opt/hipfire/.hipfire_kernels:z" in svc["volumes"]
    assert "/opt/ods/data/hipfire/models:/opt/hipfire/.hipfire/models:z" in svc["volumes"]
    assert (inst / "data" / "agent" / "kernels").is_dir()     # per_instance_dirs created


def test_multi_gpu_claim_joins_rocr_with_commas(tmp_path):
    doc = {**DOC, "kind": "lemonade", "gpu_indices": [3, 4], "env": {}}
    r, out, _ = _render(tmp_path, doc)
    assert r.returncode == 0, r.stderr
    svc = json.loads(out.read_text())["services"]["agent"]
    assert svc["environment"]["ROCR_VISIBLE_DEVICES"] == "3,4"
    assert svc["ports"] == ["127.0.0.1:11500:8080"]


def test_unknown_kind_exits_2_and_names_known_kinds(tmp_path):
    r, out, _ = _render(tmp_path, {**DOC, "kind": "nope"})
    assert r.returncode == 2 and "unknown kind 'nope'" in r.stderr and "hipfire" in r.stderr
    assert not out.exists()


def test_env_outside_allowlist_exits_1(tmp_path):
    r, out, _ = _render(tmp_path, {**DOC, "env": {"HIPFIRE_MODEL": "m", "ROCR_VISIBLE_DEVICES": "0,1"}})
    assert r.returncode == 1 and "not allowed for kind 'hipfire'" in r.stderr
    assert not out.exists()


def test_malformed_document_exits_1(tmp_path):
    docp = tmp_path / "doc.json"; docp.write_text("{nope")
    r = subprocess.run([sys.executable, str(RENDER), str(TEMPLATES), str(docp),
                        str(tmp_path / "o.yaml"), str(tmp_path), "/opt/ods"], capture_output=True, text=True)
    assert r.returncode == 1 and "unusable instance document" in r.stderr


@pytest.mark.xfail(strict=True, reason="comfyui.json lands in Task 6")
def test_every_template_in_kinds_json_exists_and_has_the_schema():
    kinds = json.loads((TEMPLATES / "kinds.json").read_text())
    for kind, fname in kinds.items():
        t = json.loads((TEMPLATES / fname).read_text())
        assert {"image", "internal_port", "service", "environment", "env_allow", "volumes",
                "per_instance_dirs", "route"} == set(t), kind
        # never render the empty-string ROCm traps
        assert "HSA_OVERRIDE_GFX_VERSION" not in t["environment"] and "HSA_XNACK" not in t["environment"]
        # the pin is always written by the renderer, never inherited
        assert "ROCR_VISIBLE_DEVICES" not in t["environment"]
