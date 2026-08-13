"""DEPRECATED alias — /api/spark/* forwards to the ONE swap node's serving
routes (app.routers.serving). One deploy cycle only (design §6); the README
notes the removal target. Resolution: exactly one control:"swap" node ->
forward; none -> 503 with the legacy unbuilt-engine message (existing
feature-detecting callers keep working); several -> 409 naming the
candidates — never guess ([[literal-declared-inputs]])."""

from fastapi import APIRouter, Request

from app.routers.serving import (
    ReloadBody,
    SwapBody,
    serving_reload,
    serving_status,
    serving_swap,
    single_swap_node_id,
)

router = APIRouter(prefix="/spark", tags=["spark"])


@router.get("/status")
def spark_status(request: Request) -> dict:
    return serving_status(single_swap_node_id(request.app.state.deck), request)


@router.post("/swap")
def spark_swap(body: SwapBody, request: Request) -> dict:
    return serving_swap(single_swap_node_id(request.app.state.deck), body, request)


@router.post("/reload")
def spark_reload(body: ReloadBody, request: Request) -> dict:
    return serving_reload(single_swap_node_id(request.app.state.deck), body, request)
