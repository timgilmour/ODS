"""
Spark router — remote single-slot swap control (lifecycle only, no VRAM
arbitration; see app/engines/spark.py for why). No auth, matching the rest
of the deck (ops-first single-operator box; LAN sits behind Authelia).

deck["spark"] is None on boxes with no spark configured — both endpoints
answer 503 then, so the UI can feature-detect with one GET. Engine
exceptions (GuardError/BusyError/EngineError) propagate to the app-wide
handlers (409/409/502), per the house policy in control.py.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/spark", tags=["spark"])


class SwapBody(BaseModel):
    profile: str
    # Skips only the busy guard; the litellm default-route guard is
    # force-proof by design (see SparkClient.swap).
    force: bool = False


def _spark(request: Request):
    spark = request.app.state.deck.get("spark")
    if spark is None:
        raise HTTPException(status_code=503,
                            detail="spark engine is not configured")
    return spark


@router.get("/status")
def spark_status(request: Request) -> dict:
    return _spark(request).status()


@router.post("/swap")
def spark_swap(body: SwapBody, request: Request) -> dict:
    out = _spark(request).swap(body.profile, force=body.force)
    return {"status": "ok", **out}
