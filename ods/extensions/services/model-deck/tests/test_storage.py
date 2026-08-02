"""Tests for app.storage.plan_move — pure guard logic."""
import pytest

from app.engines import GuardError
from app.storage import plan_move, unit_in_use


def _unit(**over):
    u = {"id": "hot:a.gguf", "type": "gguf", "name": "a.gguf", "location": "hot",
         "relpath": "a.gguf", "size": 1000, "mtime": 1.0, "state": "resident",
         "pinned": False, "last_used": None}
    u.update(over)
    return u


def _dest(**over):
    d = {"name": "cold", "path": "/stores/cold", "role": "cold", "store_type": "gguf",
         "engine": "none", "watermark_gb": None, "archive_to": None, "readonly": False,
         "uuid": "u1", "available": True, "free_bytes": 10**12, "total_bytes": 10**12}
    d.update(over)
    return d


def _world(lemonade_model=None, default_route=None, comfy_state="idle", comfy_queue=0):
    return {"gpus": [], "externals": [], "placement": {},
            "default_route": default_route,
            "tenants": {"lemonade": {"state": "loaded" if lemonade_model else "unloaded",
                                     "model": lemonade_model, "footprint": None, "idle_s": None},
                        "comfyui": {"state": comfy_state, "queue": comfy_queue, "idle_s": None},
                        "hipfire": {"state": "parked", "model": None, "footprint": 0, "queue_depth": None}}}


def _plan(unit=None, dest=None, world=None, active=frozenset(), free=None, slack=0):
    dest = dest or _dest()
    return plan_move(unit or _unit(), dest, world or _world(), active,
                     dest["free_bytes"] if free is None else free, slack)


def test_happy_path_returns_plan():
    assert _plan() == {"unit_id": "hot:a.gguf", "src_location": "hot",
                       "dest_location": "cold", "bytes": 1000}


def test_loaded_model_refused():
    with pytest.raises(GuardError, match="currently loaded"):
        _plan(world=_world(lemonade_model="extra.a.gguf"))


def test_default_route_never_movable():
    with pytest.raises(GuardError, match="default route"):
        _plan(world=_world(default_route="extra.a.gguf"))


def test_busy_comfy_unit_refused():
    unit = _unit(id="cm:loras/x.st", type="comfy", name="x.st", location="cm", relpath="loras/x.st")
    with pytest.raises(GuardError):
        _plan(unit=unit, world=_world(comfy_state="busy", comfy_queue=2))


def test_unavailable_and_moving_and_inflight_refused():
    with pytest.raises(GuardError):
        _plan(unit=_unit(state="unavailable"))
    with pytest.raises(GuardError):
        _plan(unit=_unit(state="moving"))
    with pytest.raises(GuardError, match="in flight"):
        _plan(active=frozenset({"hot:a.gguf"}))


def test_dest_guards():
    with pytest.raises(GuardError):
        _plan(dest=_dest(name="hot"))                     # same as source
    with pytest.raises(GuardError):
        _plan(dest=_dest(available=False, free_bytes=None))
    with pytest.raises(GuardError):
        _plan(dest=_dest(readonly=True))
    with pytest.raises(GuardError, match="insufficient space"):
        _plan(free=500, slack=0)
    with pytest.raises(GuardError, match="insufficient space"):
        _plan(free=1400, slack=500)                       # slack counts


def test_unit_in_use_helper():
    assert unit_in_use(_unit(), _world(lemonade_model="extra.a.gguf")) is not None
    assert unit_in_use(_unit(), _world()) is None
    assert unit_in_use(_unit(name="other.gguf"), _world(default_route="extra.other.gguf")) is not None


# storage_decide tests
from app.storage import storage_decide


def _loc(name, role="hot", free=10 * 10**9, wm=None, archive_to=None, available=True, **over):
    d = {"name": name, "path": f"/stores/{name}", "role": role, "store_type": "gguf",
         "engine": "none", "watermark_gb": wm, "archive_to": archive_to, "readonly": False,
         "uuid": "u", "available": available, "free_bytes": free if available else None,
         "total_bytes": 100 * 10**9}
    d.update(over)
    return d


def _u(uid, loc="hot", size=2 * 10**9, last_used=None, pinned=False, state="resident", name=None):
    return {"id": uid, "type": "gguf", "name": name or uid.split(":")[1], "location": loc,
            "relpath": uid.split(":")[1], "size": size, "mtime": 1.0, "state": state,
            "pinned": pinned, "last_used": last_used}


def test_healthy_watermark_no_actions():
    locs = [_loc("hot", free=60 * 10**9, wm=50.0, archive_to="cold"), _loc("cold", role="cold")]
    assert storage_decide([_u("hot:a.gguf")], locs, _world(), 0) == []


def test_lru_eviction_until_watermark_met():
    locs = [_loc("hot", free=47 * 10**9, wm=50.0, archive_to="cold"),
            _loc("cold", role="cold", free=100 * 10**9)]
    units = [_u("hot:old.gguf", last_used=100.0), _u("hot:never.gguf", last_used=None),
             _u("hot:recent.gguf", last_used=999.0)]
    actions = storage_decide(units, locs, _world(), 0)
    # needed 3 GB; never-used goes first, then LRU — two 2 GB archives suffice
    assert [a["unit_id"] for a in actions] == ["hot:never.gguf", "hot:old.gguf"]
    assert all(a["type"] == "archive" and a["dest"] == "cold" for a in actions)


def test_exemptions_pinned_loaded_default_route():
    locs = [_loc("hot", free=1 * 10**9, wm=50.0, archive_to="cold"),
            _loc("cold", role="cold", free=100 * 10**9)]
    units = [_u("hot:pinned.gguf", pinned=True),
             _u("hot:loaded.gguf"), _u("hot:route.gguf")]
    world = _world(lemonade_model="extra.loaded.gguf", default_route="extra.route.gguf")
    actions = storage_decide(units, locs, world, 0)
    assert [a["type"] for a in actions] == ["shortfall"]          # nothing eligible


def test_partial_relief_archives_then_reports_shortfall():
    locs = [_loc("hot", free=40 * 10**9, wm=50.0, archive_to="cold"),
            _loc("cold", role="cold", free=100 * 10**9)]
    units = [_u("hot:only.gguf", size=4 * 10**9)]
    actions = storage_decide(units, locs, _world(), 0)
    assert actions[0]["type"] == "archive"
    assert actions[1] == {"type": "shortfall", "location": "hot",
                          "missing_bytes": 6 * 10**9}


def test_dest_space_accounts_for_planned_archives():
    locs = [_loc("hot", free=40 * 10**9, wm=50.0, archive_to="cold"),
            _loc("cold", role="cold", free=3 * 10**9)]              # fits one 2 GB, not two
    units = [_u("hot:a.gguf", last_used=1.0), _u("hot:b.gguf", last_used=2.0)]
    actions = storage_decide(units, locs, _world(), 0)
    archives = [a for a in actions if a["type"] == "archive"]
    assert [a["unit_id"] for a in archives] == ["hot:a.gguf"]


def test_no_archive_to_reports_shortfall_only():
    locs = [_loc("hot", free=1 * 10**9, wm=50.0, archive_to=None)]
    actions = storage_decide([_u("hot:a.gguf")], locs, _world(), 0)
    assert [a["type"] for a in actions] == ["shortfall"]


# StorageWatcher tests
from app.storage import StorageWatcher


class _FakeQueue:
    def __init__(self):
        self.submitted = []
        self._active = False
    def active(self):
        return self._active
    def submit(self, plan, label, on_success=None):
        self.submitted.append((plan, label))
        return {"id": "j1", **plan}


class _Settings:
    storage_watch_interval = 60.0
    storage_slack_bytes = 0


def _watcher_env(tmp_path, auto, free_gb=1):
    from app.catalog import Catalog
    from app.locations import LocationStore
    from app.policy import StoragePolicyStore
    hot = tmp_path / "hot"; hot.mkdir()
    cold = tmp_path / "cold"; cold.mkdir()
    (hot / "idle.gguf").write_bytes(b"x" * 1000)
    locs = LocationStore(tmp_path / "locations.json",
                         disk_usage=lambda p: type("u", (), {
                             "total": 100 * 10**9, "used": 0,
                             "free": free_gb * 10**9 if str(p).endswith("hot") else 50 * 10**9})())
    locs.register({"name": "hot", "path": str(hot), "role": "hot", "store_type": "gguf",
                   "engine": "lemonade", "watermark_gb": 5.0, "archive_to": "cold", "readonly": False})
    locs.register({"name": "cold", "path": str(cold), "role": "cold", "store_type": "gguf",
                   "engine": "none", "watermark_gb": None, "archive_to": None, "readonly": False})
    cat = Catalog(tmp_path / "catalog.json", locs)
    pol = StoragePolicyStore(tmp_path / "storage_policy.json")
    pol.put({"auto": auto})
    queue = _FakeQueue()
    watcher = StorageWatcher(_Settings(), locs, cat, pol, queue,
                             world_fn=lambda: _world(), events_path=tmp_path / "events.jsonl")
    return watcher, queue, tmp_path / "events.jsonl"


def test_auto_on_enqueues_archive(tmp_path):
    watcher, queue, _ = _watcher_env(tmp_path, auto=True)
    watcher.tick()
    assert len(queue.submitted) == 1
    plan, label = queue.submitted[0]
    assert plan["unit_id"] == "hot:idle.gguf" and plan["dest_location"] == "cold"
    assert label == "watermark archive"


def test_auto_off_suggests_only(tmp_path):
    from app.events import tail_events
    watcher, queue, events = _watcher_env(tmp_path, auto=False)
    watcher.tick()
    assert queue.submitted == []
    kinds = [e["kind"] for e in tail_events(events)]
    assert "storage_suggestion" in kinds


def test_yields_to_active_jobs(tmp_path):
    watcher, queue, events = _watcher_env(tmp_path, auto=True)
    queue._active = True
    watcher.tick()
    assert queue.submitted == []


def test_suggestion_deduped_across_ticks(tmp_path):
    from app.events import tail_events
    watcher, queue, events = _watcher_env(tmp_path, auto=False)
    watcher.tick(); watcher.tick(); watcher.tick()
    kinds = [e["kind"] for e in tail_events(events)]
    assert kinds.count("storage_suggestion") == 1


def test_healthy_watermark_no_events(tmp_path):
    from app.events import tail_events
    watcher, queue, events = _watcher_env(tmp_path, auto=True, free_gb=50)
    watcher.tick()
    assert queue.submitted == [] and tail_events(events) == []


def test_thread_start_stop(tmp_path):
    watcher, _, _ = _watcher_env(tmp_path, auto=False)
    watcher.start()
    watcher.stop()          # must join promptly (interval-bounded via stop Event)
