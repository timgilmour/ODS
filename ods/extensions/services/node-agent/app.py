"""ODS node-agent: read-only metrics endpoint for remote inference nodes."""
import socket
import time

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import Response

import nodeconfig
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


def _collect_gpus_uncached() -> list[dict]:
    gpus = collect_detailed_gpus(nodeconfig.GPU_BACKEND)
    return [g.model_dump() for g in (gpus or [])]


_gpu_cache: dict = {"expires": 0.0, "value": None}


def _collect_gpus_cached() -> list[dict]:
    now = time.monotonic()
    if now < _gpu_cache["expires"] and _gpu_cache["value"] is not None:
        return _gpu_cache["value"]
    value = _collect_gpus_uncached()
    _gpu_cache["expires"] = now + nodeconfig.GPU_CACHE_TTL_SECONDS
    _gpu_cache["value"] = value
    return value


@app.get("/v1/node/info", dependencies=[Depends(verify_key)])
def node_info():
    return {
        "name": nodeconfig.NODE_NAME,
        "hostname": socket.gethostname(),
        "platform": nodeconfig.GPU_BACKEND,
        "capabilities": ["metrics"],
        "gpus": _collect_gpus_uncached(),
    }
