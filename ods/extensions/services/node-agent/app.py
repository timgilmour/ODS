"""ODS node-agent: metrics + file-protocol swap control for remote nodes."""
import secrets
import socket
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fastapi.responses import Response

import nodeconfig
import serving
import settings_store
import swapctl
from gpu_collect import collect_detailed_gpus

# docs/redoc/openapi are unauthenticated by construction, and this service runs
# with network_mode: host -- so leaving them on would advertise the whole API
# surface to anyone who can reach the port. Every real route is bearer-gated.
app = FastAPI(title="ods-node-agent", docs_url=None, redoc_url=None,
              openapi_url=None)


class AuthError(Exception):
    pass


@app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc: AuthError):
    return Response(status_code=401)


def verify_key(authorization: str = Header(default="")) -> None:
    if not nodeconfig.NODE_AGENT_KEY:
        raise AuthError()  # fail closed: no key configured, nothing is allowed
    expected = f"Bearer {nodeconfig.NODE_AGENT_KEY}"
    # Constant-time, and compared as UTF-8 bytes: compare_digest raises
    # TypeError on non-ASCII str, and this header is attacker-controlled, so a
    # str compare would turn an unauthenticated request into a 500 not a 401.
    # Matches dashboard-api/security.py and ods/bin/ods-host-agent.py.
    if not secrets.compare_digest(authorization.encode("utf-8"),
                                  expected.encode("utf-8")):
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
    capabilities = ["metrics"]
    if swapctl.enabled():
        capabilities.append("swap")
    return {
        "name": nodeconfig.NODE_NAME,
        "hostname": socket.gethostname(),
        "platform": nodeconfig.GPU_BACKEND,
        "capabilities": capabilities,
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


@app.exception_handler(swapctl.SwapCtlDisabled)
async def _swapctl_disabled(request: Request, exc: swapctl.SwapCtlDisabled):
    return JSONResponse(status_code=503,
                        content={"detail": "swap control is not configured"})


class SwapBody(BaseModel):
    profile: str


@app.get("/v1/node/profiles", dependencies=[Depends(verify_key)])
def node_profiles():
    return {"profiles": swapctl.list_profiles(),
            "swap_status": swapctl.read_status()}


@app.post("/v1/node/swap", status_code=202, dependencies=[Depends(verify_key)])
def node_swap(body: SwapBody):
    try:
        req_id = swapctl.request_swap(body.profile)
    except swapctl.InvalidProfile:
        raise HTTPException(status_code=400, detail="invalid profile name")
    except swapctl.UnknownProfile:
        raise HTTPException(status_code=404, detail="unknown profile")
    except swapctl.SwapInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"id": req_id, "profile": body.profile}


@app.exception_handler(settings_store.SettingsDisabled)
async def _settings_disabled(request: Request, exc: settings_store.SettingsDisabled):
    return JSONResponse(status_code=503,
                        content={"detail": "settings are not configured"})


@app.exception_handler(swapctl.InvalidProfile)
async def _invalid_profile(request: Request, exc: swapctl.InvalidProfile):
    return JSONResponse(status_code=400, content={"detail": "invalid profile name"})


@app.exception_handler(ValueError)
async def _settings_value_error(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/v1/node/profile/{profile}/settings", dependencies=[Depends(verify_key)])
def get_settings(profile: str):
    return settings_store.read_settings(profile)


@app.put("/v1/node/profile/{profile}/settings", dependencies=[Depends(verify_key)])
def put_settings(profile: str, document: dict):
    settings_store.write_settings(profile, document)
    return document


@app.get("/v1/node/profile/{profile}/compose", dependencies=[Depends(verify_key)])
def get_compose(profile: str):
    if not swapctl._NAME_RE.match(profile or ""):
        raise swapctl.InvalidProfile(profile)
    vllm, _ = swapctl._dirs()
    path = vllm / f"compose-{profile}.yaml"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="unknown profile")
    return {"profile": profile, "text": path.read_text()}


@app.get("/v1/node/catalog", dependencies=[Depends(verify_key)])
def get_catalog():
    return {"catalog": settings_store.read_newest_catalog()}
