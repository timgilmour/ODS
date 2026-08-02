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
