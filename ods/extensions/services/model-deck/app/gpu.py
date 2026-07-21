"""
Model Deck sysfs GPU/VRAM reader.

Pure functions over injected sysfs roots — no Settings import, no module
state, no I/O outside the given paths. In production ``drm_root``/``kfd_root``
are the container binds ``/sysdrm`` and ``/syskfd``; tests pass tmp_path
fakes.

Real-world layout modeled (from the live host):
  <drm_root>/cardN/device/mem_info_vram_total  — byte integer, trailing newline
  <drm_root>/cardN/device/mem_info_vram_used   — byte integer, trailing newline

  drm_root also contains non-GPU entries (connector dirs like card0-DP-1,
  renderD128, version, ...). Only cardN dirs that actually expose
  device/mem_info_vram_total count as GPUs; everything else is skipped.

  <kfd_root>/<pid>/vram_<gpuid>                — byte integer, trailing newline

  <gpuid> is a KFD-internal id, unrelated to the drm card index, and a pid
  dir may contain other unrelated files. Attribution of a pid's vram_*
  files to a GPU is by the order of its *nonzero* vram_* files against the
  qualifying GPU list (sorted card order) — display-only, feeds an
  "external tenants" list in the UI, and nothing evicts based on it.

Any missing or unreadable file/dir, anywhere, is skipped rather than
raised.
"""

from pathlib import Path

# Below this, a card is treated as an integrated/display GPU and excluded
# entirely — it does not consume a slot in the returned index sequence.
MIN_VRAM_BYTES = 4 * 1024**3  # 4 GiB


def _read_int(path: Path) -> int | None:
    """Read an integer from a sysfs-style file. None if missing/unreadable/malformed."""
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def _discover_cards(drm_root: Path) -> list[Path]:
    """Sorted device dirs for card* entries that actually expose VRAM sysfs.

    Skips non-GPU drm entries (connector dirs, renderD128, version, ...) —
    only dirs with a device/mem_info_vram_total file count as GPUs.
    """
    try:
        entries = sorted(drm_root.iterdir())
    except OSError:
        return []
    cards = []
    for entry in entries:
        if not entry.name.startswith("card"):
            continue
        device_dir = entry / "device"
        if (device_dir / "mem_info_vram_total").is_file():
            cards.append(device_dir)
    return cards


def _read_kfd_pids(kfd_root: Path) -> dict[int, list[int]]:
    """pid -> ordered nonzero vram_* byte values, in filename-sorted order.

    Display-only quality bar for external-tenant attribution; not an
    authoritative gpuid match.
    """
    try:
        entries = sorted(kfd_root.iterdir())
    except OSError:
        return {}

    result: dict[int, list[int]] = {}
    for pid_dir in entries:
        if not pid_dir.is_dir():
            continue
        try:
            pid = int(pid_dir.name)
        except ValueError:
            continue
        try:
            vram_files = sorted(pid_dir.glob("vram_*"))
        except OSError:
            continue
        values = [n for f in vram_files if (n := _read_int(f))]
        if values:
            result[pid] = values
    return result


def read_gpus(drm_root: Path, kfd_root: Path) -> list[dict]:
    """Read per-GPU VRAM totals/usage and external-tenant pid maps.

    Returns one dict per qualifying GPU (vram_total >= 4 GiB), indexed by
    position in sorted card* order after skipping iGPUs and non-GPU
    entries:

        {"index": int, "vram_total": int, "vram_used": int, "pids": {pid: bytes}}
    """
    gpus: list[dict] = []
    for device_dir in _discover_cards(drm_root):
        total = _read_int(device_dir / "mem_info_vram_total")
        used = _read_int(device_dir / "mem_info_vram_used")
        if total is None or used is None or total < MIN_VRAM_BYTES:
            continue
        gpus.append({"index": len(gpus), "vram_total": total, "vram_used": used, "pids": {}})

    if not gpus:
        return gpus

    for pid, values in _read_kfd_pids(kfd_root).items():
        for gpu, value in zip(gpus, values):
            gpu["pids"][pid] = value

    return gpus
