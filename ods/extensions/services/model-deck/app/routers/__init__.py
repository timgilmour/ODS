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

``build_lifecycle_view`` also stamps each entry with ``settings_drift``
(Task 7): a placement whose settings-store scopes were touched more
recently than its intent's ``last_healthy_ts``. This is a DISPLAY flag
only. ``app.reconcile.plan_reconcile`` takes ``statuses``/``intents``
directly — never this view — so a settings edit can never, by construction,
make the reconciler restart anything; conflating the two would restart a
serving model because someone typed in a settings box.
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
            "settings_drift": _settings_drift(deck, key, intent),
        }
    return view


def _settings_drift(deck: dict, key: str, intent: dict | None) -> dict | None:
    """``{"changed": [keys], "since": iso}`` when settings recorded for this
    placement were written more recently than its intent's
    ``last_healthy_ts``, else ``None``.

    ``key`` is the lifecycle key (``<node>/<resource>``, e.g.
    ``local/hipfire``); the settings-store scope key is ``<node>/<engine>``
    (app.settings_store, app.routers.settings._resolve). They coincide for
    lemonade/hipfire/comfyui but not for spark (resource ``slot0``, engine
    ``spark``), so this rebuilds the scope key from ``intent["engine"]``
    rather than assuming `key` itself is the settings key.

    ``last_healthy_ts`` is None for a placement that has never been
    confirmed healthy — there is no known-good baseline to compare against,
    so ANY settings recorded for it are treated as unconfirmed drift rather
    than silently swallowed: they might be exactly what should run, or might
    be why it never came up, and either way an operator should see them.

    A pure read: never writes, never consulted by app.reconcile.plan_reconcile
    (which takes `statuses`/`intents` directly, not this view) — settings
    drift is a flag, never a restart trigger.
    """
    if not intent:
        return None
    store = deck.get("settings_store")
    if store is None:
        return None
    engine = intent.get("engine")
    if not engine:
        return None

    node = key.split("/", 1)[0]
    engine_key = f"{node}/{engine}"
    model = intent.get("model")
    baseline = intent.get("last_healthy_ts")

    scopes = [("engines", engine_key)]
    if model:
        scopes.append(("models", model))
        scopes.append(("engine_models", f"{engine_key}|{model}"))

    changed: list[str] = []
    since: str | None = None
    for kind, scope_key in scopes:
        entry = store.scope(kind, scope_key)
        updated_ts = entry.get("updated_ts")
        if updated_ts is None:
            continue
        if baseline is not None and updated_ts <= baseline:
            continue
        for namespace in ("args", "env", "container"):
            for name in entry.get(namespace, {}):
                if name not in changed:
                    changed.append(name)
        if since is None or updated_ts > since:
            since = updated_ts

    if not changed:
        return None
    return {"changed": changed, "since": since}
