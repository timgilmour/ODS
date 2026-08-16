"""Declared remote-managed engines (engines.json) + a read-only status probe.

engines.json is host-owned and lives beside profiles.json (see swapctl.py's
_dirs() -- same NODE_VLLM_DIR). This agent only ever READS engines.json --
declarations are never written here. Actuation is a separate write:
request_engine() below drops <ctl>/engine-req.json for the host-side
swap-helper (Task 2) to execute; the agent has no docker access, so writing
that file is the entire actuation, and nothing here reads a result back
(engine-status-<resource>.json is forensics only).

There is no metrics surface on the node today, so "is this engine busy" is
answered by a declared probe rather than a scrape target. Only
``{"kind": "connections", "port": <int>}`` exists -- a bare established-TCP
count on the engine's port, read from /proc/net/tcp{,6} (works because this
agent runs `network_mode: host`, app.py:20-23). refuse-never-coerce: any
other kind is a load-time ValueError, not a silent default -- a "metrics"
kind is not built speculatively; it returns only if/when an engine actually
exposes one.
"""

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

import nodeconfig
import swapctl

# Loader is strict: exactly these keys, nothing more, nothing missing.
_ENTRY_KEYS = frozenset({"compose_file", "health_url", "busy"})
_BUSY_KEYS = frozenset({"kind", "port"})

_PROC_TCP = "/proc/net/tcp"
_PROC_TCP6 = "/proc/net/tcp6"
_ESTABLISHED = "01"  # /proc/net/tcp st field; 0A is LISTEN


@dataclass(frozen=True)
class EngineDecl:
    name: str
    compose_file: str
    health_url: str
    busy_port: int


def _validate_busy(name: str, busy) -> int:
    if not isinstance(busy, dict) or set(busy) != _BUSY_KEYS:
        raise ValueError(
            f"engine {name!r}: busy must have exactly the keys {sorted(_BUSY_KEYS)}")
    kind = busy["kind"]
    if kind != "connections":
        raise ValueError(
            f"engine {name!r}: unsupported busy.kind {kind!r} -- only "
            '"connections" exists today (no speculative kinds)')
    port = busy["port"]
    # bool is a subclass of int in Python -- refuse-never-coerce means
    # True/False must never silently pass as 1/0.
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"engine {name!r}: busy.port must be an int in 1-65535")
    return port


def _validate_entry(name: str, entry) -> EngineDecl:
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        raise ValueError(
            f"engine {name!r} must have exactly the keys {sorted(_ENTRY_KEYS)}")
    compose_file = entry["compose_file"]
    if not isinstance(compose_file, str) or not compose_file:
        raise ValueError(f"engine {name!r}: compose_file must be a non-empty string")
    if not Path(compose_file).is_absolute():
        raise ValueError(f"engine {name!r}: compose_file must be an absolute path")
    health_url = entry["health_url"]
    if not isinstance(health_url, str) or not health_url:
        raise ValueError(f"engine {name!r}: health_url must be a non-empty string")
    busy_port = _validate_busy(name, entry["busy"])
    return EngineDecl(name=name, compose_file=compose_file, health_url=health_url,
                      busy_port=busy_port)


def load_engines(path) -> dict[str, EngineDecl]:
    """Load and strictly validate engines.json at ``path``.

    A missing file means a node with no declared engines, which is normal --
    same contract as profiles.json (swapctl._profiles_meta_map) -- and
    returns {}. Anything PRESENT but malformed (bad JSON, extra/missing
    keys, an unsupported busy.kind, a relative compose_file, an
    out-of-range port, ...) is refused loudly with ValueError: unlike
    profiles.json this file gates real engine control, so refuse-never-
    coerce applies to its shape, not only its field values.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("engines.json must be a JSON object")
    return {name: _validate_entry(name, entry) for name, entry in data.items()}


def _configured_path() -> Path | None:
    """Resolve engines.json: NODE_ENGINES_FILE env override, else
    <NODE_VLLM_DIR>/engines.json beside profiles.json, else unconfigured
    (mirrors how swapctl._dirs() reads NODE_VLLM_DIR/NODE_SWAP_CTL_DIR)."""
    raw = (nodeconfig.NODE_ENGINES_FILE or "").strip()
    if raw:
        return Path(raw)
    vllm = (nodeconfig.NODE_VLLM_DIR or "").strip()
    if vllm:
        return Path(vllm) / "engines.json"
    return None


def load_configured_engines() -> dict[str, EngineDecl]:
    """load_engines() against the resolved path; {} when unconfigured."""
    return load_engines(_configured_path())


def _count_in_text(text: str, port_hex: str) -> int:
    """Pure parser: count ESTABLISHED (st==01) rows whose LOCAL port
    matches ``port_hex``.

    Each data row is `sl local_address rem_address st ...` with
    local_address formatted `<hex addr>:<hex port>`. Only the LOCAL port is
    ever compared -- a peer connecting FROM this engine's port number (it
    would show up as rem_address) must never be mistaken for load on it.
    """
    count = 0
    for line in text.splitlines()[1:]:  # skip the header row
        fields = line.split()
        if len(fields) < 4:
            continue
        local_port = fields[1].rpartition(":")[2]
        state = fields[3]
        if local_port.upper() == port_hex and state.upper() == _ESTABLISHED:
            count += 1
    return count


def count_established(port: int, tcp_path: str = _PROC_TCP,
                      tcp6_path: str = _PROC_TCP6) -> int:
    """Established-connection count on ``port``, summed over IPv4 + IPv6.

    Paths are injectable so tests exercise the real parsing logic against
    fixture text rather than monkeypatching module constants. This only
    means anything because the agent runs `network_mode: host`
    (app.py:20-23) -- these are the HOST's connection tables. Raises
    OSError if a proc file is missing/unreadable; callers (engine_status)
    decide what that means for the caller-facing result.
    """
    port_hex = f"{port:04X}"
    total = 0
    for path in (tcp_path, tcp6_path):
        total += _count_in_text(Path(path).read_text(), port_hex)
    return total


def engine_status(decl: EngineDecl, *, count_established=count_established,
                  client=httpx) -> dict:
    """Probe one declared engine: {"reachable", "healthy", "busy_requests"}.

    The connection count is taken BEFORE the health GET so the health
    probe's own connection can never be observed as load on the very next
    read. A counter failure (missing/unreadable /proc files, or a parse
    error) reports busy_requests: None, never 0 -- a silent 0 would read as
    "idle" to a caller that should instead fail toward treating an engine
    of unknown load as busy.

    ``client`` defaults to the ``httpx`` module itself (its ``.get`` module
    function matches the injected fake clients' shape); the 2.0s timeout and
    try/except httpx.HTTPError shape mirror serving.py's _fetch_raw.
    """
    try:
        busy_requests = count_established(decl.busy_port)
    except (OSError, ValueError):
        busy_requests = None

    try:
        resp = client.get(decl.health_url, timeout=2.0)
        resp.raise_for_status()
        ok = True
    except httpx.HTTPError:
        ok = False

    return {"reachable": ok, "healthy": ok, "busy_requests": busy_requests}


class EngineRequestPending(Exception):
    pass


def request_engine(name: str, verb: str) -> None:
    """Write <ctl>/engine-req.json = {"resource", "verb", "ts"} for the
    host-side swap-helper to consume (swap-helper.sh's engine up/down
    protocol comment).

    Mirrors swapctl.request_swap's file-protocol shape and atomicity
    (tmp file + rename) and reuses swapctl._dirs() to locate the ctl dir --
    the same NODE_VLLM_DIR/NODE_SWAP_CTL_DIR gate that governs profile
    swaps governs engine requests too, so an unconfigured node answers the
    same SwapCtlDisabled either way. The compose file to act on is never
    part of this payload -- only `resource` (a bare name) and `verb` -- the
    helper alone resolves `resource` against the host-owned engines.json
    allowlist, which is the real security boundary (defense in depth: this
    agent already refused to write for a `name` that isn't a declared
    engine, at the route layer, before this function is ever called).

    The request file IS the queue and its capacity is one: a pending file
    raises EngineRequestPending rather than being clobbered, mirroring
    request_swap's SwapInProgress check on request.json.
    """
    _, ctl = swapctl._dirs()
    req_path = ctl / "engine-req.json"
    if req_path.exists():
        raise EngineRequestPending(
            f"an engine request is already pending, cannot queue {verb!r} for {name!r}")
    tmp = ctl / f".engine-req.{uuid.uuid4()}.tmp"
    tmp.write_text(json.dumps({"resource": name, "verb": verb, "ts": time.time()}))
    tmp.rename(req_path)
