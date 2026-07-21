"""
Model Deck admin auth — a single FastAPI dependency gating every mutating
endpoint.

Two independent, either-sufficient ways to prove admin:

* a shared-secret header (``X-Deck-Token``) compared against
  ``settings.admin_token``. Disabled entirely when ``admin_token`` is empty
  (the default for tests and bare-uvicorn runs — see ``app.settings``), so
  an empty header can never match an empty token.
* a ``Remote-Groups`` header containing ``admins`` among its comma-split
  values. This is forward-auth injected by caddy and trusted because only
  caddy can reach this container from outside the LAN (see the compose
  network topology) — Model Deck itself does no group verification.

GETs never carry this dependency at all; every mutating endpoint does, via
``Depends(require_admin)``.
"""

from fastapi import Header, HTTPException, Request, status


def require_admin(
    request: Request,
    x_deck_token: str | None = Header(default=None, alias="X-Deck-Token"),
    remote_groups: str | None = Header(default=None, alias="Remote-Groups"),
) -> None:
    settings = request.app.state.deck["settings"]

    if settings.admin_token and x_deck_token == settings.admin_token:
        return

    if remote_groups is not None:
        groups = {group.strip() for group in remote_groups.split(",")}
        if "admins" in groups:
            return

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="admin auth required")
