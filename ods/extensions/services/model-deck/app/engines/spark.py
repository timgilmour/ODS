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
     requests (vLLM /metrics). A metrics failure while the endpoint is up
     is EngineError, not "not busy". force=True skips this guard. Engine-
     aware: only runs when the current profile's engine is "vllm" or
     unknown (conservative default); a non-vllm engine (e.g. comfyui) has
     no /metrics to consult, so the guard is skipped — v1 documented
     limitation, no ComfyUI queue visibility, the operator owns not
     swapping mid-render.
  3. boot-window guard — endpoint down + last swap "done"/"swapping" means
     a boot (possibly a ~15 min autotune) is in flight; refuse rather than
     silently restart it. force=True interrupts — which is also the
     recovery path for a profile whose boot has wedged. A last swap in
     state "error" never started a boot, so swapping away needs no force
     (found live 2026-07-30: helper "done" just means swap.sh launched).
The node-agent's own 409 (pending request / helper mid-swap) surfaces as
BusyError; its 4xx validation answers surface as EngineError.
"""

from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import httpx

from app.engines import BusyError, EngineError, GuardError
from app.engines.litellm import LiteLLMClient

_TIMEOUT = httpx.Timeout(5.0)

_BUSY_METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting")

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

    def swap_in_progress(self) -> bool:
        """True while a previous swap is still booting. Same judgement the
        boot-window guard in swap() makes, exposed for the deck's lifecycle
        reconciler; adds no node-agent endpoint."""
        return boot_in_flight(self.status())

    def busy_requests(self) -> int:
        """Sum of running+waiting requests from vLLM's /metrics."""
        try:
            resp = self._serving.get("/metrics")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        total = 0.0
        for line in resp.text.splitlines():
            if line.startswith(_BUSY_METRICS):
                try:
                    total += float(line.rsplit(None, 1)[-1])
                except ValueError:
                    raise EngineError(f"unparseable metric line: {line!r}")
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
            engine = self._current_engine()
            if engine in (None, "vllm"):
                n = self.busy_requests()
                if n > 0:
                    raise GuardError(
                        f"spark serving has {n} in-flight request(s); "
                        "retry later or use force")
            # else: non-vllm engine (e.g. comfyui) — no /metrics to consult;
            # busy guard deliberately skipped (v1 documented limitation: no
            # ComfyUI queue visibility — the operator owns not swapping
            # mid-render).
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
