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
recently than its intent's ``updated_ts`` (NOT ``last_healthy_ts`` — see
``_settings_drift``'s docstring for why that comparison self-erases). This
is a DISPLAY flag only. ``app.reconcile.plan_reconcile`` takes
``statuses``/``intents`` directly — never this view — so a settings edit can
never, by construction, make the reconciler restart anything; conflating
the two would restart a serving model because someone typed in a settings
box.
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
    from app.observe import spark_node_id

    store = deck.get("intent_store")
    if store is None:
        return {}

    intents = store.get()
    # One load of settings.json for the whole view build (Task 7 review
    # round, 2026-08-07), not one per resource per scope: _settings_drift
    # used to call SettingsStore.scope() up to 3x per lifecycle entry, each
    # a fresh file read. `.get()` already returns the same healed shape
    # `.scope()` slices from, so this is a pure hoist, not a behavior
    # change.
    settings_store = deck.get("settings_store")
    settings_data = settings_store.get() if settings_store is not None else None

    # One load of the spark profile->identity map for the whole view build
    # (Task 6), same idea as the settings_data hoist above: _settings_drift
    # only consults this for the spark slot key, but every OTHER key's call
    # still receives it (harmless — see _settings_drift's key gate), so
    # reading it once here beats a CharacteristicsStore.entry() call per
    # lifecycle resource. entry()/`.get("profile_identities")` reads as
    # ``{}``/``None`` when nothing has been adopted yet, which unwraps to
    # ``None`` below — the pre-Task-5 state, and _settings_drift's C1
    # fallback handles it identically to no map at all.
    characteristics_store = deck.get("characteristics_store")
    identity_map = None
    if characteristics_store is not None:
        identities_field = characteristics_store.entry(
            f"engine/{spark_node_id()}/vllm"
        ).get("profile_identities")
        identity_map = (identities_field or {}).get("value")

    view = {}
    for key, obs in build_observations(deck, world).items():
        intent = intents.get(key)
        view[key] = {
            **derive_status(intent, obs),
            "intent": intent,
            "observed": obs,
            "last_healthy_ts": (intent or {}).get("last_healthy_ts"),
            "settings_drift": _settings_drift(
                settings_data, key, intent, identity_map=identity_map
            ),
        }
    return view


def _settings_drift(
    settings_data: dict | None,
    key: str,
    intent: dict | None,
    identity_map: dict | None = None,
) -> dict | None:
    """``{"changed": ["namespace:key", ...], "since": iso}`` when settings
    recorded for this placement were written more recently than its
    intent's ``updated_ts``, else ``None``.

    Baseline is ``intent["updated_ts"]``, NOT ``last_healthy_ts`` — CRITICAL
    fix, Task 7 review round 2026-08-07. ``app.arbiter.Watcher._reconcile_pass``
    calls ``note_healthy(key)`` on every tick a placement is observed
    ``serving`` (app/arbiter.py, the loop right before ``plan_reconcile``),
    and ``note_healthy`` unconditionally re-stamps ``last_healthy_ts`` to
    now — so comparing against it made the flag self-erase within one
    arbiter tick of the placement actually serving, i.e. drift was visible
    only while NOT serving, backwards from the feature's purpose.
    ``updated_ts`` is stable while serving: ``IntentStore.record()`` stamps
    it at every DELIBERATE load/unload/park (operator action, set-apply, or
    the arbiter's own contention-driven load/unload — "whoever actuates,
    records", app/arbiter.py:_execute), which is exactly the moment a
    process (re)launches and starts consuming its settings; neither
    ``note_healthy`` nor a plain reconciler restore (``_execute_restore`` /
    ``_restore``, which calls the engine directly without re-recording,
    since intent already agreed) touches it.

    ``key`` is the lifecycle key (``<node>/<resource>``, e.g.
    ``local/hipfire``); the settings-store scope key is ``<node>/<engine>``
    (app.settings_store, app.routers.settings._resolve). They coincide for
    lemonade/hipfire/comfyui but not for spark (resource ``slot0``, engine
    ``spark``), so this rebuilds the scope key from ``intent["engine"]``
    rather than assuming `key` itself is the settings key.

    Spark additionally speaks TWO vocabularies at once (Task 6, 5th
    vocabulary-bug instance caught at plan time): intent records the deck
    adapter name (``engine: "spark"``) and the PROFILE (``routers/spark.py``
    deliberately records profiles — swap takes a profile, and mm27b serves
    under a different --served-model-name, so comparing served names would
    report permanent false drift). Settings, though, live under the real
    engine (``"vllm"``) and the checkpoint identity (Task 5's
    ``CharacteristicsStore`` ``profile_identities`` field maps a profile to
    its identity/service/container_name). Left untranslated, a PUT to
    ``engine_models/sparky/vllm|<identity>`` would never register against
    the verbatim scope key ``sparky/spark|heretic`` intent builds — settings
    drift silently dead for spark, the exact D11 live-drill flow. So when
    ``key`` is the spark slot and ``identity_map`` has an entry for the
    profile intent recorded, the scope list is built from the TRANSLATED
    engine (``vllm``) and identity instead of the verbatim adapter/profile.
    Every other key, and a spark call with no (matching) map entry, keeps
    C1's verbatim behavior exactly — the translation is opt-in per call, not
    a redefinition of what "engine"/"model" mean everywhere.

    No intent at all (``intent`` is ``None``) means nothing is running
    deliberately, so there is nothing a settings write could be "since" —
    ``None`` is the honest answer, not a suppressed positive.

    Each namespace of a scope entry carries its OWN ``updated_ts`` (Task 7
    review round — an entry-level clock made a written env value light up
    a same-tick-untouched args key too). ``changed`` entries are qualified
    ``"namespace:key"`` (e.g. ``"args:max-model-len"``, never a bare
    ``"max-model-len"``) so same-named keys in different namespaces stay
    distinguishable and never dedupe into one. Within a namespace whose
    stamp postdates the baseline, every CURRENT key of that namespace is
    reported — not just the key(s) a single put() actually touched, since
    this store keeps no per-key write history to diff against. Accepted
    approximation for C1; C2's set snapshots are expected to make this
    exact.

    A pure read: never writes, never consulted by app.reconcile.plan_reconcile
    (which takes `statuses`/`intents` directly, not this view) — settings
    drift is a flag, never a restart trigger.
    """
    from app.observe import SPARK_SLOT_KEY

    if not intent or settings_data is None:
        return None
    engine = intent.get("engine")
    if not engine:
        return None

    node = key.split("/", 1)[0]
    engine_key = f"{node}/{engine}"
    model = intent.get("model")

    # Spark-slot translation (Task 6) — see the docstring above. Gated on
    # the exact key AND a matching map entry so a spark call with no
    # (matching) profile_identities is byte-identical to C1: no map yet
    # adopted, or a profile that was never adopted, still resolves scopes
    # from intent verbatim rather than silently going dark.
    if key == SPARK_SLOT_KEY and identity_map and model in identity_map:
        engine_key = f"{node}/vllm"
        model = identity_map[model]["identity"]

    baseline = intent.get("updated_ts")

    scopes = [("engines", engine_key)]
    if model:
        scopes.append(("models", model))
        scopes.append(("engine_models", f"{engine_key}|{model}"))

    changed: list[str] = []
    since: str | None = None
    for kind, scope_key in scopes:
        entry = settings_data.get(kind, {}).get(scope_key, {})
        namespace_ts = entry.get("updated_ts")
        if not isinstance(namespace_ts, dict):
            continue
        for namespace in ("args", "env", "container"):
            ts = namespace_ts.get(namespace)
            if ts is None:
                continue
            if baseline is not None and ts <= baseline:
                continue
            for name in entry.get(namespace, {}) or {}:
                qualified = f"{namespace}:{name}"
                if qualified not in changed:
                    changed.append(qualified)
            if since is None or ts > since:
                since = ts

    if not changed:
        return None
    return {"changed": changed, "since": since}
