"""
Spark router — remote single-slot swap control (lifecycle only, no VRAM
arbitration; see app/engines/spark.py for why). No auth, matching the rest
of the deck (ops-first single-operator box; LAN sits behind Authelia).

deck["spark"] is None on boxes with no spark configured — both endpoints
answer 503 then, so the UI can feature-detect with one GET.

A successful swap records intent for the slot (app.intent), which is what
makes the spark a reconciled resource rather than a read-only one. Engine
exceptions (GuardError/BusyError/EngineError) propagate to the app-wide
handlers (409/409/502), per the house policy in control.py.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.observe import SPARK_SLOT_KEY

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
    deck = request.app.state.deck
    out = _spark(request).swap(body.profile, force=body.force)
    # Last, and only on success: a guard-refused swap never happened, and
    # intent is last-known-GOOD. Without this write the slot is only ever
    # read, derives 'unmanaged' forever, and the reconciler's spark restore
    # branch is unreachable in production.
    #
    # The identity recorded is the PROFILE, not the served model name:
    # mm27b serves under --served-model-name aeon, so recording the served
    # name would report permanent drift for a perfectly correct placement
    # (app.observe.translate_spark_status makes the same choice).
    deck["intent_store"].record(
        SPARK_SLOT_KEY, state="loaded", model=body.profile, engine="spark")
    # The swap just changed what the observation cache is holding.
    observer = deck.get("spark_observer")
    if observer is not None:
        observer.invalidate()
    return {"status": "ok", **out}
