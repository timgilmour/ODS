"""Tests for app.catalog.Catalog — typed unit scanner + sticky metadata."""
import pytest

from app.catalog import Catalog
from app.locations import LocationStore


def _fs(tmp_path):
    """Two locations: hot gguf store + cold hf store."""
    hot = tmp_path / "hot"; hot.mkdir()
    (hot / "a.gguf").write_bytes(b"x" * 10)
    (hot / "notes.txt").write_bytes(b"ignore me")          # non-gguf ignored
    cold = tmp_path / "cold"; cold.mkdir()
    repo = cold / "Qwen3.5-27B"; repo.mkdir()
    (repo / "config.json").write_bytes(b"{}")
    (repo / "model-00001.safetensors").write_bytes(b"y" * 20)
    store = LocationStore(tmp_path / "locations.json")
    store.register({"name": "hot", "path": str(hot), "role": "hot", "store_type": "gguf",
                    "engine": "lemonade", "watermark_gb": None, "archive_to": None, "readonly": False})
    store.register({"name": "cold", "path": str(cold), "role": "cold", "store_type": "hf",
                    "engine": "none", "watermark_gb": None, "archive_to": None, "readonly": False})
    return store, hot, cold


def test_scan_builds_typed_units(tmp_path):
    store, _, _ = _fs(tmp_path)
    cat = Catalog(tmp_path / "catalog.json", store)
    units = {u["id"]: u for u in cat.scan()}
    assert units["hot:a.gguf"]["type"] == "gguf" and units["hot:a.gguf"]["size"] == 10
    repo = units["cold:Qwen3.5-27B"]
    assert repo["type"] == "hf_repo" and repo["size"] == 22 and repo["state"] == "resident"
    assert not any(u["name"] == "notes.txt" for u in units.values())


def test_comfy_units_carry_relpath(tmp_path):
    root = tmp_path / "comfy"; (root / "loras").mkdir(parents=True)
    (root / "loras" / "style.safetensors").write_bytes(b"z" * 5)
    store = LocationStore(tmp_path / "locations.json")
    store.register({"name": "cm", "path": str(root), "role": "hot", "store_type": "comfy",
                    "engine": "comfyui", "watermark_gb": None, "archive_to": None, "readonly": False})
    units = Catalog(tmp_path / "catalog.json", store).scan()
    assert units[0]["relpath"] == "loras/style.safetensors" and units[0]["name"] == "style.safetensors"


def test_unavailable_location_retains_units(tmp_path):
    from app.locations import MARKER_NAME
    store, hot, cold = _fs(tmp_path)
    cat = Catalog(tmp_path / "catalog.json", store)
    cat.scan()
    (cold / MARKER_NAME).unlink()                      # "drive unplugged"
    units = {u["id"]: u for u in cat.scan()}
    assert units["cold:Qwen3.5-27B"]["state"] == "unavailable"   # retained, NOT vanished
    assert units["hot:a.gguf"]["state"] == "resident"


def test_pin_and_last_used_survive_rescan(tmp_path):
    store, _, _ = _fs(tmp_path)
    clock = {"t": 1000.0}
    cat = Catalog(tmp_path / "catalog.json", store, clock=lambda: clock["t"])
    cat.scan()
    cat.set_pinned("hot:a.gguf", True)
    cat.note_used_gguf("a.gguf")
    units = {u["id"]: u for u in cat.scan()}
    assert units["hot:a.gguf"]["pinned"] is True
    assert units["hot:a.gguf"]["last_used"] == 1000.0


def test_record_moved_rewrites_id_and_keeps_meta(tmp_path):
    store, hot, cold = _fs(tmp_path)
    cat = Catalog(tmp_path / "catalog.json", store)
    cat.scan()
    cat.set_pinned("hot:a.gguf", True)
    new = cat.record_moved("hot:a.gguf", "cold")
    assert new["id"] == "cold:a.gguf" and new["pinned"] is True and new["location"] == "cold"
    assert cat.get("hot:a.gguf") is None


def test_part_and_marker_files_never_scanned(tmp_path):
    store, hot, _ = _fs(tmp_path)
    (hot / "b.gguf.part").write_bytes(b"partial")
    units = Catalog(tmp_path / "catalog.json", store).scan()
    assert not any(".part" in u["relpath"] for u in units)


def test_set_pinned_unknown_id_raises(tmp_path):
    store, _, _ = _fs(tmp_path)
    cat = Catalog(tmp_path / "catalog.json", store)
    cat.scan()
    with pytest.raises(ValueError):
        cat.set_pinned("hot:nope.gguf", True)


def test_moving_state_survives_rescan(tmp_path):
    """A move marks its unit "moving" before the worker can see the job; a
    rescan landing mid-move must not reset it to resident (the storage
    watcher's candidate filter and plan_move's in-flight guard both read it)."""
    store, _, _ = _fs(tmp_path)
    cat = Catalog(tmp_path / "catalog.json", store)
    cat.scan()
    cat.set_state("hot:a.gguf", "moving")

    units = {u["id"]: u for u in cat.scan()}
    assert units["hot:a.gguf"]["state"] == "moving"


def test_catalog_operations_are_mutually_exclusive(tmp_path):
    """Every load-modify-save method shares one lock. Without it a scan can
    land between a move's source-unlink and record_moved(), erasing the unit
    and falsifying a successful move."""
    import threading

    store, _, _ = _fs(tmp_path)
    cat = Catalog(tmp_path / "catalog.json", store)
    cat.scan()

    in_write = threading.Event()
    release = threading.Event()
    result = {}
    original_save = cat._save

    def blocking_save(units):
        original_save(units)
        in_write.set()
        assert release.wait(5.0)          # keep holding whatever lock we're under

    cat._save = blocking_save
    writer = threading.Thread(target=lambda: cat.set_state("hot:a.gguf", "moving"))
    writer.start()
    assert in_write.wait(5.0)

    reader = threading.Thread(target=lambda: result.update(n=len(cat.units())))
    reader.start()
    reader.join(0.2)
    assert reader.is_alive(), "units() ran while a writer held the catalog mid-write"

    release.set()
    reader.join(5.0)
    writer.join(5.0)
    assert not reader.is_alive() and result["n"] == 2
