"""Concurrent-writer safety for every JSON store [T9b].

All of these do load-modify-save through a FIXED tmp path + os.replace.
Unlocked, two writers fail two ways at once: a stale read swallows the other's
write, AND the racing os.replace hits a tmp file the other thread already
moved — FileNotFoundError raised into whatever thread it was. On an HTTP route
that is a 500; on the arbiter's 2 s tick it is a dead pass.

Found by a sweep after the same defect turned up in CharacteristicsStore
(max-review c4), whose test in test_characteristics.py is this file's template.

DRIVEN DIRECTLY, not through TestClient. FastAPI's sync routes really do run
on separate threadpool threads, but TestClient may serialize requests — a
green test through the client would prove nothing about the store.

Intent, Characteristics and Catalog have their own concurrency tests.
Settings and Provenance are LOCKED but have no thread test anywhere — a real
coverage gap, recorded rather than papered over. Registry is deliberately absent: its
only write path (`observe()`) has zero live callers, so locking dead code
would be inventing a contract nobody uses.
"""

import threading

from app.declared import DeclaredStore
from app.locations import LocationStore
from app.node_store import NodeStore
from app.policy import PolicyStore, StoragePolicyStore
from app.sets import ConfigSet, SetStore


def _hammer(fns, rounds=40):
    """Run each callable in its own barrier-started thread; return whatever
    they raised (empty list when the store is safe)."""
    start = threading.Barrier(len(fns))
    errors = []

    def run(fn):
        try:
            start.wait(timeout=5)
            for i in range(rounds):
                fn(i)
        except Exception as exc:  # noqa: BLE001 — reported through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(fn,)) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        # A hung writer would otherwise let every assertion below pass
        # vacuously — join(timeout) returns whether or not it finished.
        assert not t.is_alive(), "a writer thread did not finish"
    return errors


def test_policy_store_survives_a_tick_racing_an_http_read(tmp_path):
    """The worst one. The arbiter tick reads policy every 2 s while HTTP
    routes read and write it, and since the task-3 boundary gate _load()
    ITSELF writes (it persists the heal) — so two concurrent READS are two
    concurrent writers on a partial file. Starting from a corrupt file puts
    both threads on the healing path at once, which is the real interleave.
    """
    path = tmp_path / "policy.json"
    path.write_text("{ not json")
    # Declare lemonade as a known resource for this test
    declared_defaults = {
        "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900}
    }
    store = PolicyStore(path, declared_defaults=lambda: declared_defaults)

    errors = _hammer([
        lambda i: store.get(),                       # the watcher tick's read
        lambda i: store.auto_enabled(),              # a status route's read
        lambda i: store.put({"lemonade": {"priority": 50 + (i % 3),
                                          "pinned": False, "idle_ttl": 900}}),
    ])

    assert errors == []
    assert store.get()["lemonade"]["pinned"] is False


def test_storage_policy_store_survives_watcher_heal_racing_a_put(tmp_path):
    path = tmp_path / "storage_policy.json"
    path.write_text("{ not json")
    store = StoragePolicyStore(path)

    errors = _hammer([
        lambda i: store.get(),                       # StorageWatcher's 60 s pass
        lambda i: store.put({"auto": bool(i % 2)}),  # the HTTP route
    ])

    assert errors == []
    assert set(store.get()) == {"auto"}


def test_declared_store_concurrent_writes_to_different_keys_all_land(tmp_path):
    """HTTP-vs-HTTP: two PUTs to DIFFERENT keys still read-modify-write the
    same file, so one silently loses without the lock."""
    store = DeclaredStore(tmp_path / "declared.json")

    errors = _hammer([
        lambda i: store.put(f"oci:local:alpha{i}", {"tools_verified": True}),
        lambda i: store.put(f"oci:local:beta{i}", {"notes": "seen"}),
    ])

    assert errors == []
    data = store.get()
    assert len([k for k in data if k.startswith("oci:local:alpha")]) == 40
    assert len([k for k in data if k.startswith("oci:local:beta")]) == 40


def test_node_store_concurrent_adds_all_land(tmp_path):
    """Two threadpool threads registering different nodes. NodeStore has TWO
    fixed tmp paths (nodes + credentials), each its own race surface — so the
    credential sidecar is asserted too, not just the topology file."""
    store = NodeStore(tmp_path / "nodes.json", tmp_path / "creds.json")

    def add(prefix):
        return lambda i: store.add(
            {"id": f"{prefix}{i}", "label": f"{prefix} {i}",
             "agent_kind": "node-agent", "address": f"http://{prefix}{i}:7720"},
            credential=f"key-{prefix}{i}")

    errors = _hammer([add("alpha"), add("beta")])

    assert errors == []
    ids = {n["id"] for n in store.list()}
    assert len([i for i in ids if i.startswith("alpha")]) == 40
    assert len([i for i in ids if i.startswith("beta")]) == 40
    assert store.credential_for("alpha39") == "key-alpha39"
    assert store.credential_for("beta39") == "key-beta39"


def test_location_store_concurrent_registers_all_land(tmp_path):
    """HTTP-vs-HTTP over one shared list file."""
    store = LocationStore(tmp_path / "locations.json")

    def register(prefix):
        def go(i):
            root = tmp_path / f"{prefix}{i}"
            root.mkdir()
            store.register({"name": f"{prefix}{i}", "path": str(root),
                            "role": "cold", "store_type": "gguf",
                            "engine": "none", "watermark_gb": None,
                            "archive_to": None, "readonly": False})
        return go

    errors = _hammer([register("alpha"), register("beta")], rounds=20)

    assert errors == []
    names = {loc["name"] for loc in store.list()}
    assert len([n for n in names if n.startswith("alpha")]) == 20
    assert len([n for n in names if n.startswith("beta")]) == 20


def test_location_store_refuses_a_duplicate_name_under_concurrency(tmp_path):
    """CRITICAL [T9b review]: register()'s duplicate-name check ran OUTSIDE
    the lock, so two concurrent registers of the SAME name both passed it and
    both appended — silent corruption of the uniqueness invariant, with
    nothing raised. Worse than a raced write, because downstream
    (routers/storage.py, routers/provenance.py) builds name-keyed dicts from
    this list, so one entry simply vanishes.

    The sibling test above cannot see it: it registers only DISTINCT names,
    so the duplicate check is never the deciding one. Exactly one thread must
    win; the other must get a ValueError.
    """
    store = LocationStore(tmp_path / "locations.json")
    root = tmp_path / "shared"
    root.mkdir()
    spec = {"name": "cold", "path": str(root), "role": "cold",
            "store_type": "gguf", "engine": "none", "watermark_gb": None,
            "archive_to": None, "readonly": False}

    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(_i):
        try:
            store.register(dict(spec))
        except ValueError:
            with lock:
                outcomes.append("refused")
        else:
            with lock:
                outcomes.append("registered")

    errors = _hammer([attempt, attempt], rounds=1)

    assert errors == []          # a ValueError here is the CORRECT outcome
    assert sorted(outcomes) == ["refused", "registered"]
    assert [loc["name"] for loc in store.list()] == ["cold"]


def test_set_store_same_slug_concurrent_saves_do_not_crash(tmp_path):
    """The narrow one: _write's tmp path carries the SLUG, so only same-name
    writers can collide.

    Two SAVES, not a save racing a delete — the first draft of this test used
    save-vs-delete and passed even with the lock removed, because os.replace
    onto a path another thread just unlinked simply recreates it. Only two
    writers contending the same `.tmp` reproduce the crash.
    """
    store = SetStore(tmp_path / "sets")

    errors = _hammer([
        lambda i: store.save(ConfigSet(name="Image session", notes=f"a{i}")),
        lambda i: store.save(ConfigSet(name="Image session", notes=f"b{i}")),
    ])

    assert errors == []
    assert store.get("image-session") is not None


def test_set_store_replace_races_a_save_on_the_same_slug(tmp_path):
    """replace() (the adopt path) writes through the same fixed per-slug tmp
    path as save(), so it needed the same lock — flagged in review as a
    residual after the first pass covered save/delete only."""
    store = SetStore(tmp_path / "sets")
    store.save(ConfigSet(name="Image session"))

    errors = _hammer([
        lambda i: store.save(ConfigSet(name="Image session", notes=f"a{i}")),
        lambda i: store.replace("image-session",
                                ConfigSet(name="Image session", notes=f"b{i}")),
    ])

    assert errors == []
    assert store.get("image-session") is not None
