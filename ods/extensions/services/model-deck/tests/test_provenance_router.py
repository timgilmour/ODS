"""Provenance HTTP surface — read, declare, desire, deep-verify, backfill.

artifact_id never travels as a path segment (it contains ':' and can contain
'/' and '~'), so every route here takes it as a query parameter or a body
field. Several tests below use ids with slashes precisely to keep that true.
"""

import pytest
from fastapi.testclient import TestClient

from app.engines import EngineError
from app.main import create_app
from app.provenance import ProvenanceStore

T0 = "2026-08-01T00:00:00+00:00"


class _FakeCatalog:
    def __init__(self, units=()):
        self._units = list(units)

    def units(self):
        return list(self._units)


class _FakeLocations:
    def __init__(self, locations=(), available=True):
        self._locations = list(locations)
        self._available = available

    def list(self):
        return list(self._locations)

    def available(self, spec):
        return self._available


class _FakeDockerCtl:
    def __init__(self, bodies=None):
        self.bodies = dict(bodies or {})

    def inspect(self, name):
        try:
            return self.bodies[name]
        except KeyError:
            raise EngineError(f"no such container {name}") from None


def _app(tmp_path, monkeypatch, *, units=(), locations=(), bodies=None,
         available=True):
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    app = create_app()
    deck = app.state.deck
    deck.update({
        "provenance_store": ProvenanceStore(
            tmp_path / "provenance.json", tmp_path / "prov-history.jsonl"),
        "provenance_history_path": tmp_path / "prov-history.jsonl",
        "catalog": _FakeCatalog(units),
        "location_store": _FakeLocations(locations, available=available),
        "dockerctl": _FakeDockerCtl(bodies) if bodies is not None else None,
    })
    return TestClient(app), deck


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    return _app(tmp_path, monkeypatch)


def _observe(deck, artifact_id="oci:local:x", version="sha256:a",
             kind="oci", node="local", role="engine", verification="exact",
             detail=None):
    deck["provenance_store"].observe(
        artifact_id, kind=kind, node=node, role=role,
        current={"version": version, "label": "v0.5.3",
                 "detail": detail or {}, "verification": verification},
        now=T0)


# --- reads ----------------------------------------------------------------

def test_list_is_empty_before_anything_is_collected(app_client):
    client, _deck = app_client
    assert client.get("/api/provenance").json() == {"artifacts": [], "gaps": []}


def test_list_reports_an_observed_artifact_as_a_gap(app_client):
    client, deck = app_client
    _observe(deck)
    body = client.get("/api/provenance").json()
    assert body["artifacts"][0]["artifact_id"] == "oci:local:x"
    assert body["gaps"] == ["oci:local:x"]     # observed, but origin unknown


# --- declared origin ------------------------------------------------------

def test_declared_origin_round_trips_and_leaves_the_gap_list(app_client):
    client, _deck = app_client
    resp = client.put("/api/provenance/origin", json={
        "artifact_id": "git:sparky:~/ds4", "role": "source",
        "origin": {"remote": "https://github.com/x/ds4", "ref": "v0.5.6",
                   "ref_kind": "tag", "default_branch": "main",
                   "diverges_from_default": True},
        "update_path": "tags live on batched-serving"})

    assert resp.status_code == 200
    body = client.get("/api/provenance").json()
    assert body["artifacts"][0]["origin"]["diverges_from_default"] is True
    assert body["artifacts"][0]["update_path"] == "tags live on batched-serving"
    assert body["gaps"] == []


def test_a_malformed_artifact_id_is_refused_not_coerced(app_client):
    client, _deck = app_client
    resp = client.put("/api/provenance/origin", json={
        "artifact_id": "nonsense", "role": "source", "origin": {}})
    assert resp.status_code == 422


def test_an_unknown_role_is_refused(app_client):
    client, _deck = app_client
    resp = client.put("/api/provenance/origin", json={
        "artifact_id": "git:sparky:~/ds4", "role": "widget", "origin": {}})
    assert resp.status_code == 422


def test_declaring_an_origin_does_not_disturb_a_measured_current(app_client):
    client, deck = app_client
    _observe(deck)
    client.put("/api/provenance/origin", json={
        "artifact_id": "oci:local:x", "role": "engine",
        "origin": {"registry": None, "repository": "x", "reference": "x:1",
                   "build": None, "archive": None}})
    assert deck["provenance_store"].entry("oci:local:x")["current"]["version"] == "sha256:a"


# --- desired --------------------------------------------------------------

def test_desired_set_then_cleared(app_client):
    client, deck = app_client
    _observe(deck)

    assert client.put("/api/provenance/desired", json={
        "artifact_id": "oci:local:x", "version": "sha256:b",
        "label": "v0.5.6"}).status_code == 200
    assert client.get("/api/provenance").json()["artifacts"][0]["version_drift"] is True

    assert client.request("DELETE", "/api/provenance/desired",
                          params={"artifact_id": "oci:local:x"}).status_code == 200
    assert client.get("/api/provenance").json()["artifacts"][0]["version_drift"] is False


def test_desired_on_an_unknown_artifact_is_404(app_client):
    client, _deck = app_client
    resp = client.put("/api/provenance/desired",
                      json={"artifact_id": "oci:local:ghost", "version": "v"})
    assert resp.status_code == 404


def test_desired_with_a_malformed_id_is_422_not_404(app_client):
    client, _deck = app_client
    resp = client.put("/api/provenance/desired",
                      json={"artifact_id": "nope", "version": "v"})
    assert resp.status_code == 422


# --- history --------------------------------------------------------------

def test_history_endpoint_returns_the_transitions(app_client):
    client, deck = app_client
    _observe(deck, version="sha256:a")
    _observe(deck, version="sha256:b")

    body = client.get("/api/provenance/history",
                      params={"artifact_id": "oci:local:x"}).json()

    assert [r["to"]["version"] for r in body["records"]] == ["sha256:a", "sha256:b"]


def test_history_of_an_unknown_artifact_is_empty_not_an_error(app_client):
    client, _deck = app_client
    resp = client.get("/api/provenance/history",
                      params={"artifact_id": "oci:local:ghost"})
    assert resp.status_code == 200
    assert resp.json()["records"] == []


# --- deep verify ----------------------------------------------------------

def test_deep_verify_hashes_the_file_and_promotes_to_exact(tmp_path, monkeypatch):
    store_dir = tmp_path / "hot"
    store_dir.mkdir()
    (store_dir / "m.gguf").write_bytes(b"weights")
    units = [{"id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "hot", "size": 7, "mtime": 2.0, "state": "resident"}]
    client, deck = _app(tmp_path, monkeypatch, units=units,
                        locations=[{"name": "hot", "path": str(store_dir)}])
    _observe(deck, artifact_id="file:local:m.gguf", version=None, kind="file",
             role="weights", verification="consistent",
             detail={"size_bytes": 7, "mtime": 2.0})

    body = client.post("/api/provenance/verify",
                       json={"artifact_id": "file:local:m.gguf"}).json()

    assert len(body["sha256"]) == 64
    assert body["matched_recorded"] is None      # nothing recorded to compare
    assert deck["provenance_store"].entry(
        "file:local:m.gguf")["current"]["verification"] == "exact"


def test_deep_verify_reports_a_changed_file_while_still_grading_it_exact(
        tmp_path, monkeypatch):
    """A mismatch is a fact about the FILE (the bytes changed), not a doubt
    about the check — we just hashed it directly, so the new sha is exact."""
    store_dir = tmp_path / "hot"
    store_dir.mkdir()
    (store_dir / "m.gguf").write_bytes(b"weights")
    units = [{"id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "hot", "size": 7, "mtime": 2.0, "state": "resident"}]
    client, deck = _app(tmp_path, monkeypatch, units=units,
                        locations=[{"name": "hot", "path": str(store_dir)}])
    _observe(deck, artifact_id="file:local:m.gguf", version="stale-sha",
             kind="file", role="weights", verification="exact",
             detail={"size_bytes": 7, "mtime": 2.0})

    body = client.post("/api/provenance/verify",
                       json={"artifact_id": "file:local:m.gguf"}).json()

    assert body["matched_recorded"] is False
    entry = deck["provenance_store"].entry("file:local:m.gguf")
    assert entry["current"]["version"] == body["sha256"]
    assert entry["current"]["verification"] == "exact"


def test_deep_verify_refuses_a_non_file_artifact(app_client):
    client, deck = app_client
    _observe(deck)
    resp = client.post("/api/provenance/verify", json={"artifact_id": "oci:local:x"})
    assert resp.status_code == 422


def test_deep_verify_on_an_unavailable_location_is_409_not_a_wrong_hash(
        tmp_path, monkeypatch):
    store_dir = tmp_path / "cold"
    store_dir.mkdir()
    units = [{"id": "cold:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "cold", "size": 7, "mtime": 2.0, "state": "unavailable"}]
    client, deck = _app(tmp_path, monkeypatch, units=units,
                        locations=[{"name": "cold", "path": str(store_dir)}],
                        available=False)
    _observe(deck, artifact_id="file:local:m.gguf", version=None, kind="file",
             role="weights", verification="unavailable",
             detail={"size_bytes": 7, "mtime": 2.0})

    resp = client.post("/api/provenance/verify",
                       json={"artifact_id": "file:local:m.gguf"})

    assert resp.status_code == 409


def test_deep_verify_on_an_unknown_artifact_is_404(app_client):
    client, _deck = app_client
    resp = client.post("/api/provenance/verify",
                       json={"artifact_id": "file:local:ghost.gguf"})
    assert resp.status_code == 404


# --- delete ---------------------------------------------------------------

def test_delete_removes_the_entry(app_client):
    client, deck = app_client
    _observe(deck)
    assert client.request("DELETE", "/api/provenance",
                          params={"artifact_id": "oci:local:x"}).status_code == 200
    assert client.get("/api/provenance").json()["artifacts"] == []


def test_an_id_with_slashes_survives_the_query_parameter_round_trip(app_client):
    client, deck = app_client
    _observe(deck, artifact_id="file:local:loras/sub/x.safetensors", version=None,
             kind="file", role="weights", verification="consistent",
             detail={"size_bytes": 1, "mtime": 2.0})

    resp = client.request("DELETE", "/api/provenance",
                          params={"artifact_id": "file:local:loras/sub/x.safetensors"})

    assert resp.status_code == 200
    assert client.get("/api/provenance").json()["artifacts"] == []


# --- /api/state -----------------------------------------------------------

def test_state_carries_a_provenance_block(app_client):
    client, deck = app_client
    _observe(deck)
    block = client.get("/api/state").json()["provenance"]
    assert block["gaps"] == 1
    assert block["drift"] == []


def test_state_drift_lists_only_real_disagreements(app_client):
    client, deck = app_client
    _observe(deck)
    client.put("/api/provenance/desired",
               json={"artifact_id": "oci:local:x", "version": "sha256:b"})
    drift = client.get("/api/state").json()["provenance"]["drift"]
    assert [d["artifact_id"] for d in drift] == ["oci:local:x"]


# --- backfill -------------------------------------------------------------

def _backfill_app(tmp_path, monkeypatch):
    units = [{"id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "hot", "size": 4096, "mtime": 2.0, "state": "resident"}]
    bodies = {"ods-hipfire": {"Image": "sha256:a",
                              "Config": {"Image": "ods-hipfire:latest"}}}
    return _app(tmp_path, monkeypatch, units=units, bodies=bodies)


def test_backfill_creates_entries_with_no_origin(tmp_path, monkeypatch):
    # Everything the derived paths can see gets an entry; NOTHING gets an
    # origin — that is the point. The gap count IS the work queue.
    client, _deck = _backfill_app(tmp_path, monkeypatch)

    body = client.post("/api/provenance/backfill").json()

    assert body["created"] == body["total"] == body["gaps"]
    assert body["total"] >= 2      # the gguf unit and the container
    assert all(a["origin"] is None
               for a in client.get("/api/provenance").json()["artifacts"])


def test_backfill_is_idempotent(tmp_path, monkeypatch):
    client, _deck = _backfill_app(tmp_path, monkeypatch)
    first = client.post("/api/provenance/backfill").json()
    second = client.post("/api/provenance/backfill").json()
    assert second["created"] == 0
    assert second["total"] == first["total"]


def test_backfill_writes_one_history_record_per_artifact_not_two(
        tmp_path, monkeypatch):
    client, _deck = _backfill_app(tmp_path, monkeypatch)
    client.post("/api/provenance/backfill")
    client.post("/api/provenance/backfill")

    for artifact in client.get("/api/provenance").json()["artifacts"]:
        records = client.get("/api/provenance/history",
                             params={"artifact_id": artifact["artifact_id"]}).json()
        assert len(records["records"]) == 1


def test_backfill_never_overwrites_a_declared_origin(tmp_path, monkeypatch):
    client, _deck = _backfill_app(tmp_path, monkeypatch)
    client.put("/api/provenance/origin", json={
        "artifact_id": "file:local:m.gguf", "role": "weights",
        "origin": {"source_kind": "hf", "repo_id": "org/model",
                   "revision": "main", "filename": "m.gguf", "url": None}})

    client.post("/api/provenance/backfill")

    entry = next(a for a in client.get("/api/provenance").json()["artifacts"]
                 if a["artifact_id"] == "file:local:m.gguf")
    assert entry["origin"]["repo_id"] == "org/model"


def test_backfill_skips_a_container_it_cannot_read(tmp_path, monkeypatch):
    client, _deck = _app(tmp_path, monkeypatch, units=(), bodies={})
    body = client.post("/api/provenance/backfill").json()
    assert body["total"] == 0      # no entry invented for an unreadable container


# --- watch (Task 9) ---------------------------------------------------------

def test_put_watch_stores_sources(app_client):
    client, _deck = app_client
    response = client.put("/api/provenance/watch", json={
        "artifact_id": "oci:local:x",
        "sources": [{"id": "t", "check": "git_tags",
                     "remote": "https://github.com/a/b",
                     "pinned": "v1.0.0", "order": "semver"}]})
    assert response.status_code == 200
    assert response.json()["watch"][0]["id"] == "t"


def test_put_watch_rejects_an_invalid_source_with_422(app_client):
    client, _deck = app_client
    response = client.put("/api/provenance/watch", json={
        "artifact_id": "oci:local:x",
        "sources": [{"id": "t", "check": "git_tags",
                     "remote": "https://github.com/a/b", "pinned": "v1"}]})
    assert response.status_code == 422


def test_put_watch_rejects_a_non_string_source_id_with_422_not_500(app_client):
    """The one-line updates.validate_watch fix: a bare truthiness check let
    a list `id` through here, and set_watch's set comprehension over source
    ids would then raise an unhandled TypeError (500) rather than the clean
    422 every other malformed watch body gets."""
    client, _deck = app_client
    response = client.put("/api/provenance/watch", json={
        "artifact_id": "oci:local:x",
        "sources": [{"id": ["not", "a", "string"], "check": "git_tags",
                     "remote": "https://github.com/a/b",
                     "pinned": "v1.0.0", "order": "semver"}]})
    assert response.status_code == 422


def test_put_watch_rejects_a_source_its_checker_cannot_execute_with_422(app_client):
    """Verified reachable in the final review: this exact body answered 200
    and was written to disk, then reported a permanent `unavailable` with
    note "checker raised KeyError" -- check_channel indexes
    `source["repository"]` (app/updates/oci.py:60). README's "Needs" column
    published it as a contract; nothing enforced it."""
    client, _deck = app_client
    response = client.put("/api/provenance/watch", json={
        "artifact_id": "oci:local:x",
        "sources": [{"id": "x", "check": "oci_channel",
                     "pinned": "sha256:aa", "order": None}]})
    assert response.status_code == 422
    assert client.get("/api/provenance").json()["artifacts"] == []


def test_put_watch_rejects_duplicate_source_ids_with_422(app_client):
    """`record_update` merges on the id (app/provenance.py:511), so the
    second source silently suppressed the first one's verdict."""
    client, _deck = app_client
    response = client.put("/api/provenance/watch", json={
        "artifact_id": "oci:local:x",
        "sources": [{"id": "same", "check": "git_tags",
                     "remote": "https://github.com/a/b",
                     "pinned": "v1.0.0", "order": "semver"},
                    {"id": "same", "check": "git_tags",
                     "remote": "https://github.com/c/d",
                     "pinned": "v2.0.0", "order": "semver"}]})
    assert response.status_code == 422
    assert client.get("/api/provenance").json()["artifacts"] == []


def test_put_watch_with_a_malformed_artifact_id_is_422_even_with_no_sources(app_client):
    """set_watch's own equality guard no-ops before ever validating the id
    when sources is `[]` and nothing is stored -- this route validates the
    id itself first, the same upfront check put_origin/put_desired already
    make, so a malformed id is refused regardless of what else is in the
    body."""
    client, _deck = app_client
    response = client.put("/api/provenance/watch", json={
        "artifact_id": "nonsense", "sources": []})
    assert response.status_code == 422


def test_put_watch_with_empty_sources_on_a_never_seen_artifact_does_not_crash(app_client):
    """set_watch is a true no-op for `sources: []` against an id with no
    stored entry (nothing to compare, nothing written) -- store.entry(...)
    then returns None, and the brief's literal `...["watch"]` would raise a
    TypeError the caller sees as an ISE. The honest answer is 'nothing is
    watched', the same shape a stored empty watch list would report."""
    client, _deck = app_client
    response = client.put("/api/provenance/watch", json={
        "artifact_id": "oci:local:never-seen", "sources": []})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "watch": []}


def test_put_watch_declared_source_survives_a_derivable_origin(app_client):
    """End-to-end: declaring an origin the collector could derive a channel
    watch from must not clobber a source an operator already hand-declared
    via this route (declared-over-derived, exercised through the HTTP
    surface rather than the pure function)."""
    client, deck = app_client
    client.put("/api/provenance/origin", json={
        "artifact_id": "oci:local:x", "role": "engine",
        "origin": {"registry": "ghcr.io", "repository": "a/b",
                   "reference": "ghcr.io/a/b:slim@sha256:dd"}})
    client.put("/api/provenance/watch", json={
        "artifact_id": "oci:local:x",
        "sources": [{"id": "upstream", "check": "git_compare",
                     "remote": "https://github.com/a/b", "ref": "main",
                     "pinned": "abc", "order": None}]})
    assert deck["provenance_store"].entry("oci:local:x")["watch"] == [
        {"id": "upstream", "check": "git_compare",
         "remote": "https://github.com/a/b", "ref": "main",
         "pinned": "abc", "order": None}]


# --- check (Task 9) ----------------------------------------------------------

class _FakeChecker:
    def __init__(self, result=None):
        self._result = result
        self.calls = 0

    def tick(self):
        self.calls += 1
        return self._result


def test_post_check_runs_a_pass_and_reports_counts(app_client):
    client, deck = app_client
    checker = _FakeChecker({"checked": 2, "available": 1})
    deck["update_checker"] = checker
    response = client.post("/api/provenance/check")
    assert response.status_code == 200
    assert response.json() == {"checked": 2, "available": 1}
    assert checker.calls == 1


def test_post_check_normalizes_a_disabled_checkers_none_to_zero_counts(app_client):
    """UpdateChecker.tick() returns None when disabled or storeless -- the
    route must still answer the documented {checked, available} shape."""
    client, deck = app_client
    deck["update_checker"] = _FakeChecker(None)
    response = client.post("/api/provenance/check")
    assert response.status_code == 200
    assert response.json() == {"checked": 0, "available": 0}


def test_post_check_without_a_registered_checker_is_a_clean_error(app_client):
    """Risk 4: MODEL_DECK_NO_WATCHER=1 (this fixture's default, like every
    other test in this file) never populates deck['update_checker'] --
    app.main only builds one when the watcher starts. `deck.get(...)` must
    yield a clean, documented error here, never a bare KeyError."""
    client, _deck = app_client
    response = client.post("/api/provenance/check")
    assert response.status_code == 503


# --- /api/state updates count (Task 9) --------------------------------------

def test_state_updates_count_reflects_a_recorded_available_status(app_client):
    client, deck = app_client
    _observe(deck)
    deck["provenance_store"].set_watch("oci:local:x", [
        {"id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1.0.0", "order": "semver"}])
    deck["provenance_store"].record_update("oci:local:x", [
        {"id": "t", "status": "available", "current": "v1.0.0",
         "latest": "v1.1.0", "detail": {}, "note": None}])

    block = client.get("/api/state").json()["provenance"]
    assert block["updates"] == 1
