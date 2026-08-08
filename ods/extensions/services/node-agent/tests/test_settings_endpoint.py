"""Settings document + compose + catalog routes.

The Deck ships a settings DOCUMENT; this node decides how to launch with it
(the swap-helper renders it into a compose override). That boundary is the
whole reason these endpoints take JSON rather than a rendered compose file.
PUT replaces rather than merges: the Deck owns ladder resolution, and a
node-side merge would give the same key two owners.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import nodeconfig
import settings_store
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}

DOC = {"args": {"max-model-len": "131072"}, "env": {}, "argv": ["serve", "/model"],
       "service": "aeon-vllm"}


def _enable(monkeypatch, tmp_path):
    monkeypatch.setattr(nodeconfig, "NODE_SETTINGS_DIR", str(tmp_path))
    return tmp_path


def test_put_persists_and_get_round_trips(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)

    put = client.put("/v1/node/profile/heretic/settings", json=DOC, headers=AUTH)
    got = client.get("/v1/node/profile/heretic/settings", headers=AUTH)

    assert put.status_code == 200
    assert got.json() == DOC
    assert json.loads((tmp_path / "heretic.json").read_text()) == DOC


def test_get_unknown_profile_returns_empty_document_not_404(monkeypatch, tmp_path):
    """'No settings yet' is the normal starting state of every profile."""
    _enable(monkeypatch, tmp_path)

    body = client.get("/v1/node/profile/never-configured/settings", headers=AUTH).json()

    assert body == {"args": {}, "env": {}, "argv": [], "service": None}


def test_put_replaces_rather_than_merges(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    client.put("/v1/node/profile/heretic/settings",
               json={**DOC, "args": {"a": "1", "b": "2"}}, headers=AUTH)

    client.put("/v1/node/profile/heretic/settings",
               json={**DOC, "args": {"b": "9"}}, headers=AUTH)

    body = client.get("/v1/node/profile/heretic/settings", headers=AUTH).json()
    assert body["args"] == {"b": "9"}


def test_unknown_top_level_key_is_422(monkeypatch, tmp_path):
    """Only the document crosses this boundary. A 'volumes' key arriving
    here would mean the Deck had started managing the node's mounts."""
    _enable(monkeypatch, tmp_path)

    resp = client.put("/v1/node/profile/heretic/settings",
                      json={**DOC, "volumes": ["/a:/b"]}, headers=AUTH)

    assert resp.status_code == 422


def test_traversal_profile_name_is_rejected(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)

    resp = client.put("/v1/node/profile/..%2Fpasswd/settings", json=DOC, headers=AUTH)

    assert resp.status_code in (400, 404)
    assert list(tmp_path.iterdir()) == []


def test_permission_error_does_not_read_as_empty_document(monkeypatch, tmp_path):
    """Regression: a mis-mounted NODE_SETTINGS_DIR (the UID/GID bug class
    this stack has actually hit) must not read as 'profile has no settings
    yet' -- that would misdirect debugging away from a broken mount. Unlike
    the catalog files, settings.json is written atomically by this same
    module, so there is no legitimate half-written case to excuse masking a
    read failure here.
    """
    _enable(monkeypatch, tmp_path)
    (tmp_path / "heretic.json").write_text(json.dumps(DOC))

    def _deny(self, *a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_text", _deny)

    local_client = TestClient(app, raise_server_exceptions=False)
    resp = local_client.get("/v1/node/profile/heretic/settings", headers=AUTH)

    # No handler is registered for a bare OSError, so this is Starlette's
    # unhandled-exception response (plain text, not JSON) -- the point of
    # the test is that it is NOT the 200 empty-document response.
    assert resp.status_code == 500
    assert resp.headers["content-type"] != "application/json"


def test_settings_dir_unset_answers_503(monkeypatch):
    monkeypatch.setattr(nodeconfig, "NODE_SETTINGS_DIR", "")

    resp = client.get("/v1/node/profile/heretic/settings", headers=AUTH)

    assert resp.status_code == 503


def test_all_new_routes_require_auth(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)

    assert client.get("/v1/node/profile/x/settings").status_code == 401
    assert client.put("/v1/node/profile/x/settings", json=DOC).status_code == 401
    assert client.get("/v1/node/profile/x/compose").status_code == 401
    assert client.get("/v1/node/catalog").status_code == 401


def test_compose_route_serves_raw_text(monkeypatch, tmp_path):
    vllm = tmp_path / "vllm"; vllm.mkdir()
    (vllm / "compose-heretic.yaml").write_text("services:\n  aeon-vllm: {}\n# a comment\n")
    monkeypatch.setattr(nodeconfig, "NODE_VLLM_DIR", str(vllm))
    monkeypatch.setattr(nodeconfig, "NODE_SWAP_CTL_DIR", str(tmp_path))

    body = client.get("/v1/node/profile/heretic/compose", headers=AUTH).json()

    assert body["profile"] == "heretic"
    assert "# a comment" in body["text"]


def test_compose_route_404_on_unknown_profile(monkeypatch, tmp_path):
    vllm = tmp_path / "vllm"; vllm.mkdir()
    monkeypatch.setattr(nodeconfig, "NODE_VLLM_DIR", str(vllm))
    monkeypatch.setattr(nodeconfig, "NODE_SWAP_CTL_DIR", str(tmp_path))

    assert client.get("/v1/node/profile/nope/compose", headers=AUTH).status_code == 404


def test_catalog_returns_newest_by_harvested_ts(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    (tmp_path / "catalog-heretic.json").write_text(json.dumps(
        {"image_id": "sha256:old", "harvested_ts": "2026-08-07T01:00:00Z",
         "engine": "vllm", "probe_output": "OLD"}))
    (tmp_path / "catalog-laguna.json").write_text(json.dumps(
        {"image_id": "sha256:new", "harvested_ts": "2026-08-07T02:00:00Z",
         "engine": "vllm", "probe_output": "NEW"}))

    body = client.get("/v1/node/catalog", headers=AUTH).json()

    assert body["catalog"]["image_id"] == "sha256:new"


def test_catalog_none_yet_is_null_not_error(monkeypatch, tmp_path):
    """Pre-first-swap is the normal state, not a failure."""
    _enable(monkeypatch, tmp_path)

    assert client.get("/v1/node/catalog", headers=AUTH).json() == {"catalog": None}


def test_catalog_skips_corrupt_files(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    (tmp_path / "catalog-bad.json").write_text("{half a json")
    (tmp_path / "catalog-good.json").write_text(json.dumps(
        {"image_id": "sha256:g", "harvested_ts": "2026-08-07T01:00:00Z",
         "engine": "vllm", "probe_output": "OK"}))

    assert client.get("/v1/node/catalog", headers=AUTH).json()["catalog"]["image_id"] == "sha256:g"


def test_catalog_is_stamped_with_the_profile_from_its_filename(monkeypatch, tmp_path):
    """The body carries engine and harvested_ts but not the profile, so the
    Deck could only attribute a digest to a profile by guessing. The FILENAME
    already knows — stamp it, and the guess becomes unnecessary."""
    _enable(monkeypatch, tmp_path)
    (tmp_path / "catalog-laguna.json").write_text(json.dumps(
        {"image_id": "sha256:new", "harvested_ts": "2026-08-08T00:00:00Z",
         "engine": "vllm", "probe_output": "{}"}))

    body = client.get("/v1/node/catalog", headers=AUTH).json()

    assert body["catalog"]["profile"] == "laguna"


def test_catalog_profile_stamp_does_not_overwrite_an_explicit_one(monkeypatch, tmp_path):
    """If the helper ever starts writing the field itself, the file wins."""
    _enable(monkeypatch, tmp_path)
    (tmp_path / "catalog-laguna.json").write_text(json.dumps(
        {"image_id": "sha256:new", "harvested_ts": "2026-08-08T00:00:00Z",
         "engine": "vllm", "probe_output": "{}", "profile": "explicit"}))

    body = client.get("/v1/node/catalog", headers=AUTH).json()

    assert body["catalog"]["profile"] == "explicit"
