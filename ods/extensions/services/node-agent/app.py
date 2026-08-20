"""ODS node-agent: metrics + file-protocol swap control for remote nodes."""
import secrets
import socket
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fastapi.responses import Response

import engines
import instances
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

# Once per process: a blind serving config (vllm profiles, no probe URL)
# announces itself in the startup log too, not only in probe() results.
serving.log_startup_warning()


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
    if instances.enabled():
        capabilities.append("instances")
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


@app.exception_handler(instances.InstancesDisabled)
async def _instances_disabled_handler(request: Request, exc: instances.InstancesDisabled):
    return JSONResponse(status_code=503,
                        content={"detail": "instance control not enabled on this node"})


class SwapBody(BaseModel):
    profile: str


@app.get("/v1/node/profiles", dependencies=[Depends(verify_key)])
def node_profiles():
    return {"profiles": swapctl.list_profiles(),
            "swap_status": swapctl.read_status()}


@app.get("/v1/node/engines", dependencies=[Depends(verify_key)])
def node_engines():
    # A node with no declared engines is normal (mirrors profiles.json):
    # {"engines": []}, not a 503 -- unlike swap control/settings, this
    # feature has no "disabled" state, only an empty one.
    return {"engines": sorted(engines.load_configured_engines())}


@app.get("/v1/node/engine/{name}/status", dependencies=[Depends(verify_key)])
def node_engine_status(name: str):
    decl = engines.load_configured_engines().get(name)
    if decl is None:
        raise HTTPException(status_code=404, detail="unknown engine")
    return engines.engine_status(decl)


def _node_engine_request(name: str, verb: str) -> dict:
    # Undeclared-resource 404 is checked here, against engines.json, before
    # request_engine() ever touches the ctl dir: the agent only writes
    # requests for declared resources (the helper re-validates `resource`
    # against its own copy of engines.json independently -- defense in
    # depth, app.py never trusts this check to be the only one).
    if name not in engines.load_configured_engines():
        raise HTTPException(status_code=404, detail="unknown engine")
    try:
        engines.request_engine(name, verb)
    except engines.EngineRequestPending as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"accepted": True}


@app.post("/v1/node/engine/{name}/up", status_code=202,
          dependencies=[Depends(verify_key)])
def node_engine_up(name: str):
    return _node_engine_request(name, "up")


@app.post("/v1/node/engine/{name}/down", status_code=202,
          dependencies=[Depends(verify_key)])
def node_engine_down(name: str):
    return _node_engine_request(name, "down")


class InstanceBody(BaseModel):
    verb: str
    document: dict


@app.post("/v1/node/instance/{resource}", status_code=202,
          dependencies=[Depends(verify_key)])
def node_instance_request(resource: str, body: InstanceBody):
    doc = instances.validate_document(body.document)  # ValueError -> 422 via the existing handler
    if doc["resource"] != resource:
        raise HTTPException(status_code=422,
                            detail=f"path resource {resource!r} != document resource {doc['resource']!r}")
    try:
        instances.request_instance(body.verb, doc)
    except instances.InstanceRequestPending as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"accepted": True}


@app.get("/v1/node/instance/{resource}/status", dependencies=[Depends(verify_key)])
def node_instance_status(resource: str):
    if not instances.NAME_RE.match(resource):
        raise HTTPException(status_code=422, detail="resource must match ^[a-z0-9][a-z0-9-]*$")
    return {"result": instances.read_status(resource)}


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
