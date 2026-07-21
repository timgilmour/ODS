"""
Model Deck admin auth — a single FastAPI dependency gating every mutating
endpoint.

Two independent, either-sufficient ways to prove admin:

* a shared-secret header (``X-Deck-Token``), timing-safe-compared against
  ``settings.admin_token``. Disabled entirely when ``admin_token`` is empty
  (the default for tests and bare-uvicorn runs — see ``app.settings``), so
  an empty header can never match an empty token.
* a ``Remote-Groups`` header containing ``admins`` among its comma-split
  values, PLUS a second shared-secret header (``X-Deck-Proxy-Key``),
  timing-safe-compared against ``settings.proxy_key``. Remote-Groups is
  forward-auth injected by caddy, but the header alone is forgeable by
  anything on the compose network that can reach this container directly
  (sibling containers) — the proxy key proves the request actually came
  through caddy rather than being spoofed. Disabled entirely when
  ``proxy_key`` is empty (the default for tests and bare-uvicorn runs).

GETs never carry this dependency at all; every mutating endpoint does, via
``Depends(require_admin)``.
"""

import hmac

from fastapi import Header, HTTPException, Request, status


def require_admin(
    request: Request,
    x_deck_token: str | None = Header(default=None, alias="X-Deck-Token"),
    remote_groups: str | None = Header(default=None, alias="Remote-Groups"),
    x_deck_proxy_key: str | None = Header(default=None, alias="X-Deck-Proxy-Key"),
) -> None:
    settings = request.app.state.deck["settings"]

    if settings.admin_token and hmac.compare_digest(x_deck_token or "", settings.admin_token):
        return

    if settings.proxy_key and hmac.compare_digest(x_deck_proxy_key or "", settings.proxy_key):
        if remote_groups is not None:
            groups = {group.strip() for group in remote_groups.split(",")}
            if "admins" in groups:
                return

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="admin auth required")
