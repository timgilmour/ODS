"""Tests for app.gpu — the sysfs GPU/VRAM reader.

Builds fake sysfs trees under tmp_path (drm + kfd) rather than touching the
real host filesystem. See app/gpu.py module docstring for the layout being
modeled.
"""

from pathlib import Path

from app.gpu import read_gpus


def _make_card(drm_root: Path, name: str, total: int, used: int | None = None) -> Path:
    """Create <drm_root>/<name>/device/mem_info_vram_{total,used} and return the device dir."""
    device_dir = drm_root / name / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "mem_info_vram_total").write_text(f"{total}\n")
    if used is not None:
        (device_dir / "mem_info_vram_used").write_text(f"{used}\n")
    return device_dir


def test_read_gpus_reports_totals_and_excludes_igpu(tmp_path):
    drm = tmp_path / "sysdrm"
    kfd = tmp_path / "syskfd"  # left nonexistent — kfd_root missing must degrade gracefully
    _make_card(drm, "card0", 34208743424, 33034301440)
    _make_card(drm, "card1", 34208743424, 33034301440)
    _make_card(drm, "card2", 2147483648, 2000000000)  # 2 GiB iGPU, below the 4 GiB floor

    # non-GPU drm noise that must be skipped without error
    (drm / "card0-DP-1").mkdir(parents=True)  # connector dir, no device/
    (drm / "renderD128").write_text("")  # not a card* dir at all
    (drm / "version").write_text("drm 1.1.0\n")

    gpus = read_gpus(drm, kfd)

    assert gpus == [
        {"index": 0, "vram_total": 34208743424, "vram_used": 33034301440, "pids": {}},
        {"index": 1, "vram_total": 34208743424, "vram_used": 33034301440, "pids": {}},
    ]


def test_read_gpus_attributes_kfd_pids_by_nonzero_order(tmp_path):
    drm = tmp_path / "sysdrm"
    kfd = tmp_path / "syskfd"
    _make_card(drm, "card0", 34208743424, 33034301440)
    _make_card(drm, "card1", 34208743424, 33034301440)

    # pid 4985 touches both GPUs — attribution is by order of its nonzero
    # vram_* files, not by the KFD-internal gpuid value.
    proc_multi = kfd / "4985"
    proc_multi.mkdir(parents=True)
    (proc_multi / "vram_1").write_text("1073741824\n")  # 1st nonzero -> gpu index 0
    (proc_multi / "vram_2").write_text("2147483648\n")  # 2nd nonzero -> gpu index 1
    (proc_multi / "vram_5").write_text("0\n")  # zero — must not consume a slot
    (proc_multi / "other_file").write_text("noise\n")  # unrelated file — ignored

    # pid 6001 only touches one GPU
    proc_single = kfd / "6001"
    proc_single.mkdir(parents=True)
    (proc_single / "vram_9").write_text("500000000\n")

    gpus = read_gpus(drm, kfd)

    assert gpus[0]["pids"] == {4985: 1073741824, 6001: 500000000}
    assert gpus[1]["pids"] == {4985: 2147483648}
    assert 6001 not in gpus[1]["pids"]


def test_read_gpus_skips_missing_and_malformed_gracefully(tmp_path):
    drm = tmp_path / "sysdrm"
    kfd = tmp_path / "syskfd"

    # card0: malformed total -> the whole card is skipped, not raised
    bad_device = drm / "card0" / "device"
    bad_device.mkdir(parents=True)
    (bad_device / "mem_info_vram_total").write_text("not-an-int\n")
    (bad_device / "mem_info_vram_used").write_text("123\n")

    # card1: valid, so it becomes the sole (index 0) GPU
    _make_card(drm, "card1", 34208743424, 33034301440)

    # kfd noise: non-numeric pid dir, a stray non-dir entry, and an
    # unreadable vram_* file — all must degrade gracefully.
    (kfd / "not-a-pid").mkdir(parents=True)
    (kfd / "not-a-pid" / "vram_1").write_text("999\n")

    (kfd / "stray_file").write_text("noise\n")

    unreadable_proc = kfd / "7777"
    unreadable_proc.mkdir(parents=True)
    unreadable_file = unreadable_proc / "vram_1"
    unreadable_file.write_text("999999\n")
    unreadable_file.chmod(0o000)

    try:
        gpus = read_gpus(drm, kfd)
    finally:
        unreadable_file.chmod(0o644)  # restore so tmp_path cleanup can remove it

    assert gpus == [
        {"index": 0, "vram_total": 34208743424, "vram_used": 33034301440, "pids": {}},
    ]


def test_read_gpus_handles_missing_roots(tmp_path):
    missing_drm = tmp_path / "no-such-drm"
    missing_kfd = tmp_path / "no-such-kfd"

    assert read_gpus(missing_drm, missing_kfd) == []


def test_read_gpus_orders_by_sorted_card_name(tmp_path):
    drm = tmp_path / "sysdrm"
    kfd = tmp_path / "syskfd"
    # Create card1 on disk before card0 to prove index order follows the
    # sorted name, not filesystem creation order.
    _make_card(drm, "card1", 34208743424, 33034301440)
    _make_card(drm, "card0", 17179869184, 1000000000)

    gpus = read_gpus(drm, kfd)

    assert [g["index"] for g in gpus] == [0, 1]
    assert gpus[0]["vram_total"] == 17179869184  # card0 sorts first
    assert gpus[1]["vram_total"] == 34208743424  # card1 sorts second
