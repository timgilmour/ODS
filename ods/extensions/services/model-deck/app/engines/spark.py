"""Spark (remote DGX) swap client — single-slot engine, lifecycle only.

The Spark serves exactly one vLLM model at a time on its own box, so there
is no VRAM arbitration to do here and this client is deliberately NOT part
of the World snapshot or the arbiter's math. It exists so the deck is the
one control plane: see the current profile, list the available ones, and
swap — through the node-agent's file-protocol swap control (the agent has
no docker access; a host-side helper validates and runs swap.sh).

Guard order in swap(), mirroring hipfire.py's park():
  1. litellm guard — refuse if litellm's `default` route points at the
     Spark's serving endpoint; a swap would break every caller that never
     asked for a spark model by name. EngineError from the route-table
     read propagates unchanged (can't see the table -> don't swap).
     force NEVER skips this guard.
  2. busy guard — refuse while the live model has running or waiting
     requests (the serving engine's /metrics). A metrics failure while the
     endpoint is up is EngineError, not "not busy". force=True skips this
     guard. Engine-aware: which metric lines count as "busy" is looked up
     per engine in _BUSY_METRICS (vllm's two gauges, ds4's single
     ds4_requests_inflight gauge — see profiles.json's "engine" field);
     unknown engine (None) is treated as "vllm" — conservative default, run
     the probe when unsure. An engine with no entry in _BUSY_METRICS (e.g.
     comfyui) has no known /metrics to consult, so the guard is skipped —
     v1 documented limitation, no ComfyUI queue visibility, the operator
     owns not swapping mid-render. All engines share one /metrics fetch —
     the node serves one profile at a time on one port (serving_url) — so
     only the metric names differ per engine, not the scrape URL. A 200
     scrape that matches NONE of the engine's expected prefixes is "0 busy"
     (fail-open) for vLLM, unchanged since launch — except for engines in
     _STRICT_BUSY_ENGINES (ds4), where it is EngineError instead: a
     misconfigured scrape or a renamed gauge must not silently read as
     "idle" and let a swap kill an in-flight generation with no signal
     anywhere (Tim's ruling, 2026-08-04).
  3. boot-window guard — endpoint down + last swap "done"/"swapping" means
     a boot (possibly a ~15 min autotune) is in flight; refuse rather than
     silently restart it. force=True interrupts — which is also the
     recovery path for a profile whose boot has wedged. A last swap in
     state "error" never started a boot, so swapping away needs no force
     (found live 2026-07-30: helper "done" just means swap.sh launched).
The node-agent's own 409 (pending request / helper mid-swap) surfaces as
BusyError; its 4xx validation answers surface as EngineError.
"""

import json
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import httpx

from app.engines import BusyError, EngineError, GuardError
from app.engines.litellm import LiteLLMClient

_TIMEOUT = httpx.Timeout(5.0)

# Per-engine busy-metric line prefixes, matched against /metrics via
# str.startswith(). New engine with a /metrics gauge -> new entry here; an
# engine with no entry (e.g. comfyui) has the busy guard skipped entirely
# (see swap()'s guard-order docstring above).
_BUSY_METRICS: dict[str, tuple[str, ...]] = {
    "vllm": ("vllm:num_requests_running", "vllm:num_requests_waiting"),
    "ds4": ("ds4_requests_inflight",),
}

# Engines where a 200 /metrics scrape matching NONE of _BUSY_METRICS[engine]
# is EngineError (loud refusal) rather than "0 busy" (silent allow, vLLM's
# behavior since launch and unchanged here). ds4-only, per Tim's 2026-08-04
# ruling: a misconfigured or upstream-renamed gauge must not let a swap kill
# an in-flight generation with no signal anywhere. New engine = new entry;
# an engine absent from this set keeps the fail-open default (matches vLLM's
# untouched behavior — the plan's "conservative path unchanged" mandate).
#
# Safe for ds4 specifically: ds4_requests_inflight is rendered from a
# zero-initialized *static* global registry (ds4.c:755 `static ds4_metrics
# g_metrics;`, comment at ds4.c:748 "Zero-initialized global"), unconditionally
# emitted by send_metrics() on every /metrics call (ds4_server.c:13421-13423)
# — present (reading 0) from the first successful scrape of a fresh,
# zero-traffic boot, never lazily created on first request. Verified
# read-only against ds4 tag v0.5.3 / commit 4ad370b4a338 on sparky
# (~/ds4), 2026-08-04.
_STRICT_BUSY_ENGINES = frozenset({"ds4"})

# swap_status states that mean a boot may still be under way. "done" is in
# here because the helper reports "done" as soon as swap.sh *launched*, not
# when the model is serving (found live 2026-07-30); "error" never started a
# boot at all.
_BOOTING_STATES = ("swapping", "done")

# How long after a swap was launched a down endpoint may still be called
# "booting". Generously above the worst observed boot: a first, uncached
# FlashInfer autotune ran ~13-15 min before the 2026-08-02 cache mounts cut
# warm boots to ~5 min.
_BOOT_WINDOW_MAX_S = 20 * 60


def boot_in_flight(status: dict, now: datetime | None = None) -> bool:
    """Whether a swap is still booting, judged from one ``status()`` payload.

    All THREE conditions are required, and each drops a different false
    reading:

    * the last swap must be in a booting state — an "error" swap never
      started a boot at all;
    * the serving endpoint must be down — a live endpoint is the boot having
      finished;
    * the swap must be recent. This is the one that is easy to miss:
      ``swap_status`` stays ``"done"`` forever after a successful swap, so
      state+endpoint alone would call a model that died hours later "still
      booting" and shield it from restore permanently — the exact 26-hour
      hipfire failure, reproduced on the spark.

    A missing or unparseable timestamp reads as still booting: unsure means
    do not act, and guessing wrong costs a multi-minute swap.
    """
    swap_status = status.get("swap_status") or {}
    if swap_status.get("state") not in _BOOTING_STATES:
        return False
    if (status.get("serving") or {}).get("endpoint_ok"):
        return False
    return _recent(swap_status.get("ts"), now)


def _recent(ts: str | None, now: datetime | None = None) -> bool:
    if not isinstance(ts, str) or not ts:
        return True
    try:
        started = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (now or datetime.now(UTC)) - started < timedelta(seconds=_BOOT_WINDOW_MAX_S)


class SparkClient:
    def __init__(
        self,
        node_url: str,
        node_key: str,
        serving_url: str,
        litellm: LiteLLMClient,
        node_transport: httpx.BaseTransport | None = None,
        serving_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._node = httpx.Client(
            base_url=node_url.rstrip("/"),
            headers={"Authorization": f"Bearer {node_key}"},
            timeout=_TIMEOUT,
            transport=node_transport,
        )
        self._serving = httpx.Client(
            base_url=serving_url.rstrip("/"),
            timeout=_TIMEOUT,
            transport=serving_transport,
        )
        self._serving_host = urlparse(serving_url).netloc
        self._litellm = litellm

    # -- reads ------------------------------------------------------------

    def _node_get(self, path: str) -> dict:
        try:
            resp = self._node.get(path)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()

    def status(self) -> dict:
        profiles = self._node_get("/v1/node/profiles")
        serving = self._node_get("/v1/node/serving")
        return {
            "profiles": profiles.get("profiles", []),
            "swap_status": profiles.get("swap_status"),
            "serving": serving,
        }

    def models(self) -> dict:
        """The serving endpoint's OpenAI-style /v1/models body — the deck's
        one source of truth for a remote model's live facts (context length,
        etc.), consumed by the characteristics derive pass
        (app.arbiter.Watcher._derive_pass / app.derive_live)."""
        try:
            resp = self._serving.get("/v1/models")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            # A 200 with a non-JSON body (misconfigured proxy, HTML error
            # page, truncated response, ...) must not escape as a raw
            # JSONDecodeError — the derive pass's
            # ``except (EngineError, BusyError)`` catch wouldn't see it.
            raise EngineError(f"non-JSON response from /v1/models: {exc}") from exc

    def get_compose(self, profile: str) -> str:
        """The raw compose YAML text node-agent holds for `profile` (GET
        /v1/node/profile/{profile}/compose) — the adopt sweep's one source
        of a profile's real launch configuration (app.compose_import,
        app.routers.settings' adopt route)."""
        return self._node_get(f"/v1/node/profile/{profile}/compose")["text"]

    def get_settings(self, profile: str) -> dict:
        """`profile`'s settings document (GET /v1/node/profile/{profile}/
        settings) — the node-agent's file-protocol store, schema {args, env,
        argv, service} (node-agent/settings_store.py's EMPTY/_KEYS)."""
        return self._node_get(f"/v1/node/profile/{profile}/settings")

    def put_settings(self, profile: str, document: dict) -> None:
        """Ship a settings document for `profile` (PUT /v1/node/profile/
        {profile}/settings) — the node-settings configure mech's write path
        (app.configure.apply_settings). Mirrors swap()'s POST below:
        EngineError on a transport failure or a non-2xx response. The
        node-agent echoes the document back on success; this client ignores
        the body — the caller already knows what it sent."""
        try:
            resp = self._node.put(f"/v1/node/profile/{profile}/settings", json=document)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)

    def swap_in_progress(self) -> bool:
        """True while a previous swap is still booting. Same judgement the
        boot-window guard in swap() makes, exposed for the deck's lifecycle
        reconciler; adds no node-agent endpoint."""
        return boot_in_flight(self.status())

    def busy_requests(self, engine: str) -> int:
        """Sum of in-flight values from ``engine``'s /metrics.

        ``engine`` is a _BUSY_METRICS key (e.g. "vllm" or "ds4") — the node
        serves one profile at a time on one port (serving_url), so the
        scrape URL is the same regardless of engine; only which metric
        lines count as "busy" changes.

        A 200 scrape that matches none of the engine's expected prefixes is
        "0 busy" (fail-open) unless ``engine`` is in _STRICT_BUSY_ENGINES,
        in which case it is EngineError (fail-closed) — see that set's
        comment for why ds4 is the one engine that needs this.
        """
        metric_prefixes = _BUSY_METRICS[engine]
        try:
            resp = self._serving.get("/metrics")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        total = 0.0
        matched = False
        for line in resp.text.splitlines():
            if line.startswith(metric_prefixes):
                matched = True
                try:
                    total += float(line.rsplit(None, 1)[-1])
                except ValueError:
                    raise EngineError(f"unparseable metric line: {line!r}")
        if not matched and engine in _STRICT_BUSY_ENGINES:
            raise EngineError(
                f"/metrics matched none of {metric_prefixes!r} for engine "
                f"{engine!r}; refusing to treat an unreadable busy signal "
                "as idle")
        return int(total)

    # -- swap -------------------------------------------------------------

    def _default_route_targets_spark(self) -> bool:
        for entry in self._litellm.model_info():
            if entry["model_name"] == "default":
                api_base = entry["litellm_params"].get("api_base", "")
                return self._serving_host in api_base
        return False

    def _current_engine(self) -> str | None:
        """Engine of the profile the node last swapped to, or None if unknown.

        Conservative by design: any ambiguity (no swap_status, a failed
        swap, or a profiles-fetch failure) returns None, which the busy
        guard in swap() treats the same as "vllm" — i.e. it still probes.
        """
        try:
            payload = self._node_get("/v1/node/profiles")
        except EngineError:
            return None
        status = payload.get("swap_status") or {}
        if status.get("state") == "error":
            return None
        current = status.get("profile")
        for prof in payload.get("profiles") or []:
            if isinstance(prof, dict) and prof.get("name") == current:
                return prof.get("engine")
        return None

    def swap(self, profile: str, force: bool = False) -> dict:
        if self._default_route_targets_spark():
            raise GuardError(
                "litellm's default route targets the spark serving endpoint; "
                "swapping would break default-route callers (force does not "
                "override this)")

        serving = self._node_get("/v1/node/serving")
        if serving.get("endpoint_ok") and not force:
            engine = self._current_engine() or "vllm"
            if engine in _BUSY_METRICS:
                n = self.busy_requests(engine)
                if n > 0:
                    raise GuardError(
                        f"spark serving has {n} in-flight request(s); "
                        "retry later or use force")
            # else: engine has no entry in _BUSY_METRICS (e.g. comfyui) —
            # no /metrics to consult; busy guard deliberately skipped (v1
            # documented limitation: no ComfyUI queue visibility — the
            # operator owns not swapping mid-render).
        if not serving.get("endpoint_ok") and not force:
            last = self._node_get("/v1/node/profiles").get("swap_status") or {}
            if last.get("state") in ("swapping", "done"):
                raise GuardError(
                    f"previous swap ({last.get('profile')}) is still booting "
                    "(a first boot can autotune ~15 min); wait for the "
                    "endpoint or use force to interrupt it")

        try:
            resp = self._node.post("/v1/node/swap", json={"profile": profile})
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if resp.status_code == 409:
            raise BusyError(resp.text)
        if not resp.is_success:
            raise EngineError(resp.text)
        body = resp.json()
        return {"id": body.get("id"), "profile": profile}
