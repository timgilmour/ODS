"""
Model Deck HTTP routers — one module per resource area, all mounted under
``/api`` by ``app.main.create_app()``:

  status.py — GET /state, GET /events (no auth)
  control.py — POST /tenants/{lemonade,comfyui,hipfire}/... (admin)
  sets.py — config-set CRUD + preview/apply (mixed: GETs open, mutations admin)
  policy.py — GET/PUT policy (GET open, PUT admin)
  storage.py — locations/catalog/moves/pins/tiering policy (no auth)
  lifecycle.py — POST /lifecycle/{quarantine clear, adopt} (the two operator
      escapes from automation)

Every router pulls its dependencies from ``request.app.state.deck`` (see
``app.main._build_deck``) rather than constructing clients itself, so tests
can swap any entry for a fake after ``create_app()`` returns.

``build_world_snapshot`` is the one helper shared across routers that need a
real-time ``World`` snapshot (status, sets preview/apply): it always
re-reads GPUs via ``deck["read_gpus"]`` and re-snapshots through the shared
``deck["world"]`` instance — never a cached/stale one — so the arbiter's
idle clocks and the HTTP surface stay in lockstep.

``build_observations`` and ``build_lifecycle_view`` are the same idea for
lifecycle state: derived on every call from intent x observation, so there
is no cached status to go stale, and shared so the status block and the
adopt route can never disagree about what is running.
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


def build_observations(deck: dict, world: dict) -> dict[str, dict]:
    """Every resource the deck can see, in app.observe's one shape.

    The spark half goes through deck["spark_observer"] — one TTL-cached,
    backed-off probe shared with the watcher, because an unreachable sparky
    is its normal state and each probe costs two 5 s timeouts.
    """
    from app.observe import merge_observations, observe_local, observe_spark

    observer = deck.get("spark_observer")
    spark_status = observer.status() if observer is not None else None
    return merge_observations(observe_local(world), observe_spark(spark_status))


def build_lifecycle_view(deck: dict, world: dict) -> dict:
    """Per-resource {status, reason, intent, observed, last_healthy_ts}.

    Read-only and derived on every call — there is no cached status to go
    stale. ``last_healthy_ts`` is lifted out of the intent record because it
    is what turns "down" into "down since when", which is the difference
    between a glance telling you something and telling you nothing.
    """
    from app.lifecycle import derive_status

    store = deck.get("intent_store")
    if store is None:
        return {}

    intents = store.get()
    view = {}
    for key, obs in build_observations(deck, world).items():
        intent = intents.get(key)
        view[key] = {
            **derive_status(intent, obs),
            "intent": intent,
            "observed": obs,
            "last_healthy_ts": (intent or {}).get("last_healthy_ts"),
        }
    return view
