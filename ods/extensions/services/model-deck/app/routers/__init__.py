"""
Model Deck HTTP routers — one module per resource area, all mounted under
``/api`` by ``app.main.create_app()``:

  status.py — GET /state, GET /events (no auth)
  control.py — POST /tenants/{lemonade,comfyui,hipfire}/... (admin)
  sets.py — config-set CRUD + preview/apply (mixed: GETs open, mutations admin)
  policy.py — GET/PUT policy (GET open, PUT admin)

Every router pulls its dependencies from ``request.app.state.deck`` (see
``app.main._build_deck``) rather than constructing clients itself, so tests
can swap any entry for a fake after ``create_app()`` returns.

``build_world_snapshot`` is the one helper shared across routers that need a
real-time ``World`` snapshot (status, sets preview/apply): it always
re-reads GPUs via ``deck["read_gpus"]`` and re-snapshots through the shared
``deck["world"]`` instance — never a cached/stale one — so the arbiter's
idle clocks and the HTTP surface stay in lockstep.
"""


def build_world_snapshot(deck: dict) -> dict:
    gpus = deck["read_gpus"](deck["drm_root"], deck["kfd_root"])
    return deck["world"].snapshot(
        gpus,
        deck["lemonade"],
        deck["comfy"],
        deck["hipfire"],
        deck["litellm"],
        deck["registry"],
    )
