"""ODS node-agent: read-only metrics endpoint for remote inference nodes."""
import socket
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import Response

import nodeconfig
import serving
from gpu_collect import collect_detailed_gpus

app = FastAPI(title="ods-node-agent")


class AuthError(Exception):
    pass


@app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc: AuthError):
    return Response(status_code=401)


def verify_key(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {nodeconfig.NODE_AGENT_KEY}"
    if not nodeconfig.NODE_AGENT_KEY or authorization != expected:
        raise AuthError()


def _collect_gpus_uncached() -> tuple[list[dict], Optional[str]]:
    """Collect one GPU sample as ``(gpus, error)``.

    ``collect_detailed_gpus()`` returns ``None`` when the collector itself is
    absent or failed, which is a different fact from a node that genuinely has
    zero GPUs (``[]``). Flattening both to ``[]`` left the dashboard showing a
    reachable node with an empty, unexplained card body, so the two stay
    distinct and a collector failure is reported as a message the dashboard
    can display verbatim.
    """
    gpus = collect_detailed_gpus(nodeconfig.GPU_BACKEND)
    if gpus is None:
        return [], (
            f"GPU collector unavailable: no usable '{nodeconfig.GPU_BACKEND}' "
            "collector on this node (check that the vendor SMI tool is "
            "installed and that the GPU devices are exposed to this container)"
        )
    return [g.model_dump() for g in gpus], None


_gpu_cache: dict = {"expires": 0.0, "value": None}


def _collect_gpus_cached() -> tuple[list[dict], Optional[str]]:
    now = time.monotonic()
    if now < _gpu_cache["expires"] and _gpu_cache["value"] is not None:
        return _gpu_cache["value"]
    value = _collect_gpus_uncached()
    _gpu_cache["expires"] = now + nodeconfig.GPU_CACHE_TTL_SECONDS
    _gpu_cache["value"] = value
    return value


@app.get("/v1/node/info", dependencies=[Depends(verify_key)])
def node_info():
    gpus, _error = _collect_gpus_uncached()
    return {
        "name": nodeconfig.NODE_NAME,
        "hostname": socket.gethostname(),
        "platform": nodeconfig.GPU_BACKEND,
        "capabilities": ["metrics"],
        "gpus": gpus,
    }


@app.get("/v1/node/gpu", dependencies=[Depends(verify_key)])
def node_gpu():
    # `error` is additive and nullable: an older dashboard-api simply ignores
    # it, a current one maps it onto the node's status card while the node
    # itself stays "online" (it answered, so it is reachable).
    gpus, error = _collect_gpus_cached()
    return {"backend": nodeconfig.GPU_BACKEND, "gpus": gpus, "error": error}


@app.get("/v1/node/serving", dependencies=[Depends(verify_key)])
def node_serving():
    return serving.probe()
