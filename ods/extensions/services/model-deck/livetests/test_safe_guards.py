"""S18–S21: expected-refusal probes. Each runs ONLY when the live state
guarantees refusal (else skip) — a safe-tier test must never actually park
or free anything by accident. NEVER send a hipfire completion here (single-slot
conversation-cache eviction — the 5.5-min re-prefill trap)."""

import pytest

from clients import hipfire_health_status
from conftest import HIPFIRE_URL

pytestmark = pytest.mark.safe


def test_s18_comfy_free_refused_while_busy(deck, comfy_direct):
    if comfy_direct.queue_len() == 0:
        pytest.skip("opportunistic: comfy queue empty — refusal not guaranteed")
    resp = deck.post("/api/tenants/comfyui/free")
    assert resp.status_code == 409
    assert "queue" in resp.json()["detail"].lower()


def test_s19_park_refused_while_hipfire_is_default_route(deck, litellm_direct):
    hip_model = deck.get("/api/state").json()["world"]["tenants"]["hipfire"]["model"]
    default = litellm_direct.route_table().get("default", "")
    if not hip_model or not default.endswith(hip_model):
        pytest.skip("opportunistic: default route does not target hipfire")
    resp = deck.post("/api/tenants/hipfire/park")
    assert resp.status_code == 409
    assert "default route" in resp.json()["detail"]
    assert hipfire_health_status(HIPFIRE_URL) == "up", "guard tripped but hipfire went down!"


def test_s20_park_refused_while_request_in_flight(deck):
    qd = deck.get("/api/state").json()["world"]["tenants"]["hipfire"].get("queue_depth")
    if not qd:
        pytest.skip("opportunistic: no hipfire request in flight — refusal not guaranteed")
    resp = deck.post("/api/tenants/hipfire/park")
    assert resp.status_code == 409
    assert hipfire_health_status(HIPFIRE_URL) == "up"


def test_s21_unknown_set_maps_to_404(deck):
    assert deck.get("/api/sets/deck-drill-no-such-slug").status_code == 404
    assert deck.post("/api/sets/deck-drill-no-such-slug/apply").status_code == 404
