"""
Docker control client — socket-proxy wrapper for container park/resume/exec.

Talks to the Docker Engine API through a tecnativa/docker-socket-proxy
sidecar (default http://docker-ctl:2375) over a 5 s httpx.Client. The proxy
accepts unversioned paths, so no `/v1.4x` prefix is needed. A `transport=`
kwarg lets tests inject httpx.MockTransport instead of touching the network.

`allowlist` is OUR enforcement, independent of (and in addition to) the
proxy's own API-class narrowing: stop()/start()/exec_run() raise GuardError,
naming the rejected container, if `name` is not in `allowlist` — checked
BEFORE any HTTP call is made, so a disallowed name never reaches the socket
at all. running()/image_ref() are reads and are NOT gated — same posture as
running() always had.

Real wire shapes this is coded against:
  GET  /containers/{name}/json       -> 200 {"State": {"Running": bool, ...},
                                              "Image": "sha256:...", ...}
                                         404 if the container doesn't exist
  POST /containers/{name}/stop?t=5   -> 204 (stopped) or 304 (already stopped)
  POST /containers/{name}/start      -> 204 (started) or 304 (already running)
  POST /containers/{name}/exec       -> 201 {"Id": "<exec_id>"}
  POST /exec/{id}/start              -> 200, multiplexed stdout/stderr stream

running()/stop()/start()/image_ref()/exec_run() raise EngineError on any
other non-2xx response (with the response text) or on an httpx.TransportError.
304 is treated as success (the container was already in the requested
state), not an error.

stop() passes `?t=5` (a 5 s SIGKILL grace period, down from Docker's 10 s
default) plus a per-request extended read timeout — LIVE-VERIFIED:
ods-llama-server ignores SIGTERM, so Docker's default 10 s grace + SIGKILL
took ~11 s end-to-end, longer than this client's 5 s default timeout,
which raised EngineError client-side while the container kept stopping
regardless. 5 s grace is safe here: the deck only ever stops idle/parked
engines (notify fires only when no model is loaded; hipfire park is a
deliberate stop), never a container mid-request. start()/running() keep
the plain 5 s client default — same idiom as LemonadeClient.load().

exec_run() also uses the extended timeout (app.harvest's probe imports vLLM
inside the container, which is not instant) and demuxes Docker's exec/start
response: with Tty=False (used unconditionally here — see exec_run's
docstring for why), Docker ALWAYS frames stdout/stderr into 8-byte-header
chunks regardless of which streams were attached, and only the stdout
frames are kept (see _demux_stdout).

DEPLOY NOTE: the socket-proxy sidecar's API surface is allowlisted by
explicit per-method path regexes in the compose command (see compose.yaml).
exec_run() needs `POST .../containers/{name}/exec` and
`POST /exec/{id}/start` allowed in addition to the existing GET .../json and
POST .../{start,stop} — a proxy still running the older allowlist will 403
every exec_run()/probe() call (image_ref() needs nothing new: it's the same
GET .../json running() already uses). The create-exec rule is pinned to the
one container harvested in C1 (`ods-hipfire`), NOT wildcarded like the
other rules — exec runs an arbitrary command as root inside whatever
container it names, so a wildcard there would make this client's in-process
`_guard` the only thing standing between a deck bug and host-wide RCE. See
compose.yaml's docker-ctl service for the two added -allowPOST lines and
its comment for the full reasoning.
"""

import httpx

from app.engines import EngineError, GuardError

_TIMEOUT = 5.0
_STOP_GRACE_S = 5
_STOP_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
# exec_run() waits on a vLLM import inside the container -- give it the same
# extended read timeout as stop()'s SIGKILL grace period, for the same
# reason (the plain 5 s client default is too short for real engine work).
_EXEC_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
_ALREADY_DONE = 304
# Docker exec/start's stream-type byte (see _demux_stdout): 1 = stdout,
# 2 = stderr. Only stdout is kept -- app.harvest's probe redirects the
# engine's own logging to stderr itself before printing its JSON, so
# anything on stderr here is expected engine noise, not payload (see
# app.harvest's module docstring, finding 3).
_STDOUT_STREAM = 1


class DockerCtl:
    def __init__(
        self,
        base_url: str,
        allowlist: list[str],
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._allowlist = allowlist
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=_TIMEOUT,
            transport=transport,
        )

    def running(self, name: str) -> bool:
        return self._inspect(name)["State"]["Running"]

    def image_ref(self, name: str) -> str:
        """The RESOLVED image content ID (`"Image"`, e.g. `"sha256:..."`)
        container `name` was created from — a cheap proxy for "engine
        version" via the same GET .../json read running() uses (no exec,
        no new proxy allowlist entry).

        Deliberately the top-level `Image` field, NOT `Config.Image`:
        `Config.Image` is the reference the container was CREATED WITH —
        for hipfire that's the floating tag `ods-hipfire:latest`
        (ods/extensions/services/hipfire/compose.amd.yaml), which is the
        SAME STRING before and after an image rebuild + container recreate.
        Keying "engine version" on it means a rebuilt vLLM behind the same
        tag would read as unchanged forever, silently skipping every
        re-harvest (caught in review before this ever shipped — see
        app.arbiter.Watcher._harvest_catalogs' docstring). The top-level
        `Image` is Docker's own resolved content ID for whatever is
        actually running right now, so it changes on every rebuild, for a
        floating tag exactly as much as a digest pin.

        This is an opaque identity for change detection, not a
        human-readable version string — see
        Watcher._harvest_catalogs' docstring and app.harvest's
        `engine_version` field.
        """
        return self._inspect(name)["Image"]

    def _inspect(self, name: str) -> dict:
        try:
            resp = self._client.get(f"/containers/{name}/json")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()

    def stop(self, name: str) -> None:
        self._guard(name)
        self._lifecycle_post(
            name,
            "stop",
            params={"t": _STOP_GRACE_S},
            timeout=_STOP_TIMEOUT,
        )

    def start(self, name: str) -> None:
        self._guard(name)
        self._lifecycle_post(name, "start")

    def exec_run(self, name: str, interpreter: str, source: str) -> str:
        """Run `interpreter -c source` inside the running container `name`
        and return its stdout (see the module docstring for why stderr is
        discarded here, not upstream).

        Two Docker Engine API calls: POST .../exec creates the exec
        instance, POST /exec/{id}/start actually runs it. Tty is False
        unconditionally — a pty would merge stdout/stderr into one stream
        BEFORE app.harvest's own stderr redirect ever gets a chance to
        separate them, defeating the whole point of that redirect (see
        app.harvest's module docstring, finding 3).

        Guarded by the same park allowlist as stop()/start(): running an
        arbitrary command inside a container is at least as powerful a
        primitive as stopping/starting it, and this client must never be
        able to do either to a container an operator hasn't explicitly
        allowed the deck to control.
        """
        self._guard(name)
        exec_id = self._exec_create(name, interpreter, source)
        return self._exec_start(exec_id)

    def _exec_create(self, name: str, interpreter: str, source: str) -> str:
        try:
            resp = self._client.post(
                f"/containers/{name}/exec",
                json={
                    "Cmd": [interpreter, "-c", source],
                    "AttachStdout": True,
                    "AttachStderr": True,
                    "Tty": False,
                },
            )
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()["Id"]

    def _exec_start(self, exec_id: str) -> str:
        try:
            resp = self._client.post(
                f"/exec/{exec_id}/start",
                json={"Detach": False, "Tty": False},
                timeout=_EXEC_TIMEOUT,
            )
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return _demux_stdout(resp.content)

    def _guard(self, name: str) -> None:
        if name not in self._allowlist:
            raise GuardError(f"container {name!r} is not in the park allowlist")

    def _lifecycle_post(self, name: str, action: str, **kwargs) -> None:
        try:
            resp = self._client.post(f"/containers/{name}/{action}", **kwargs)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if resp.status_code == _ALREADY_DONE or resp.is_success:
            return
        raise EngineError(resp.text)


def _demux_stdout(data: bytes) -> str:
    """Split Docker's exec/start response into its stdout half.

    With Tty=False, Docker always frames the stream into 8-byte-header
    chunks — 1 stream-type byte, 3 reserved zero bytes, a 4-byte big-endian
    length, then that many bytes of payload — regardless of which streams
    were attached. Frames whose type isn't _STDOUT_STREAM (stderr, in
    practice) are dropped. A truncated trailing frame (fewer than 8 header
    bytes, or a declared length longer than what remains) simply stops the
    scan rather than raising — matches this module's EngineError-on-clear-
    failure posture; a malformed frame isn't a transport error.
    """
    out = bytearray()
    i = 0
    while i + 8 <= len(data):
        stream_type = data[i]
        size = int.from_bytes(data[i + 4:i + 8], "big")
        if i + 8 + size > len(data):
            break  # declared length overruns what's actually present
        chunk = data[i + 8:i + 8 + size]
        i += 8 + size
        if stream_type == _STDOUT_STREAM:
            out.extend(chunk)
    return out.decode("utf-8", errors="replace")


class DockerEngineExec:
    """Adapts DockerCtl to Watcher's harvest contract —
    ``engine_exec(node, engine, interpreter, source) -> (version, output)``,
    with an optional ``.version`` peek — for ONE locally containerised
    engine (see app.arbiter.Watcher._harvest_catalogs and
    _configurable_engines).

    ``.version`` is a property, not a stored value: every read does a
    fresh ``image_ref`` (a GET, not an exec) so the watcher can cheaply
    check whether the running image has changed before paying for the
    exec + probe that ``__call__`` performs. A failed peek (engine down,
    proxy unreachable — EngineError; or an inspect body missing the
    expected shape — KeyError) degrades to None rather than raising: the
    peek is billed best-effort, and the watcher falls through to the
    real, still-safe call/compare path instead of crashing the derive
    pass over an optimization.

    One instance per engine, bound to that engine's container name at
    construction — Watcher._configurable_engines() names exactly one
    engine in C1 (hipfire), so ``.version`` doesn't need to be routed by
    (node, engine) yet. A second configured engine would need this to
    become a per-engine lookup (e.g. keyed by `engine`) instead of a bare
    property; noted here as a known limitation, not built ahead of need.
    """

    def __init__(self, dockerctl: DockerCtl, container: str) -> None:
        self._dockerctl = dockerctl
        self._container = container

    @property
    def version(self) -> str | None:
        try:
            return self._dockerctl.image_ref(self._container)
        except (EngineError, KeyError):
            return None

    def __call__(self, node: str, engine: str, interpreter: str, source: str) -> tuple[str, str]:
        version = self._dockerctl.image_ref(self._container)
        output = self._dockerctl.exec_run(self._container, interpreter, source)
        return version, output
