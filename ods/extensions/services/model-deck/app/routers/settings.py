"""Settings router — the two surfaces over one store.

``GET/PUT /settings/{kind}/{key}`` is the store. ``GET
/settings/effective/...`` resolves the ladder and returns TWO views of it,
not one, as of Plan C2 Task 7 (design decision 3): ``resolved`` is the full,
five-layer structured view — every key, both derived layers
(``engine_defaults``, ``checkpoint_recommendations``) included, each with
its ``{value, origin, layer}`` provenance — and ``argline`` is that same
resolution's DECLARED-ONLY subset (``origin == "declared"``, i.e. the
``engine``/``model``/``engine_model`` layers) rendered as a command line.
The two are no longer "the same data rendered twice": a harvested engine
default or a checkpoint's recommended sampling value shows up in
``resolved`` for the UI to display with its provenance, but never as a flag
on ``argline`` — an engine's own applied behavior is not something the Deck
re-asserts back at it, and this is also the exact set of layers
``app.routers.spark.spark_reload`` ships to the node (see
``_declared_only`` below; the two call sites share one filter so they can
never disagree). The optional ``?layers=`` query param (comma-separated,
validated against ``app.ladder.LAYERS``, unknown name -> 422) filters
``resolved`` ALONE, for a UI that wants to show one layer at a time;
``argline`` and ``warnings`` are always computed from the full resolution
regardless of this filter, so it can never change what would actually
ship. ``POST /settings/preview`` parses typed text without saving, which is
what makes the text field feel live.

Nothing here applies settings to a running engine: a save changes intent,
and reload stays a human click.

An absent catalog returns ``null`` rather than 404. An engine that has never
been up simply has no harvested options yet, and that is a supported state —
every flag is then 'unvalidated' and still fully editable.

Route registration order matters: the generic ``GET/PUT
/settings/{kind}/{key:path}`` routes are declared LAST. FastAPI/Starlette
matches routes in registration order, and ``{key:path}`` is greedy enough to
swallow ``catalog/sparky/vllm`` or ``effective/sparky/vllm/Qwen...`` as a
literal ``key`` if it were registered first — so ``catalog``, ``effective``,
and ``preview`` must all come before the generic pair.

``PUT`` never validates and never blocks on a warning — see
app.validate_settings. The one save-time rejection that exists,
``SettingsStore.put``'s container-namespace allowlist, raises ``ValueError``
and is deliberately NOT caught here: app.main installs a global
``ValueError`` -> 422 handler (settings-shape rejection, same family as
app.routers.facts' declared-key validation), so this router does not
duplicate that mapping. A ``PUT`` body missing ``namespace``/``values``
raises the same ``ValueError`` -> 422 rather than subscripting the body
and letting a ``KeyError`` 500 — same error family, not a special case.
``GET`` rejects an unknown ``kind`` the same way (optional fix, final
branch review 2026-08-07) — before this, a typo'd ``kind`` silently read
back ``200 {}``, indistinguishable from a real, empty scope of a valid
one.

``GET``/``PUT /settings/{kind}/{key}`` strip ``updated_ts`` out of the
returned scope before it reaches a caller: it is write-tracking bookkeeping
for ``app.routers._settings_drift``'s comparison, stamped by
``SettingsStore.put``, not a documented field of this response shape — and
leaking it would make it part of the API's contract by accident.

Engine-default decoding — F2, CRITICAL (final branch review, 2026-08-07):
``app.harvest`` stores each option's ``default`` as ``repr(action.default)``
— the probe's raw truth, e.g. ``"'auto'"`` for the Python string ``"auto"``,
``"False"``/``"None"``/``"[]"``/``"{}"`` for their respective values. Those
are text, not the values they represent; consuming them as-is (with only a
``not in (None, "None")`` filter, which only catches the ``None`` repr) fed
``"'auto'"`` — nested quotes and all — into a resolved setting and the
rendered argline, and turned every off-by-default flag (``store_true``
defaults to ``False``, ``append`` defaults to ``[]``) into a flag-with-a-
string-value that isn't even the right value. ``_decode_harvested_default``
undoes the ``repr()`` with ``ast.literal_eval`` (falling back to the raw
string on a literal it can't parse — a repr this module can't decode is
still a string value, and dropping it would lose real information a
scalar-string default wouldn't) and then applies the honest reading of each
decoded shape: ``True`` is a bare flag; ``False``/``None``/``[]``/``{}`` are
an ABSENT flag (an off-by-default option's honest rendering is no flag at
all, not the flag holding its own "off" value); everything else is a normal
scalar/list value bound for ``normalize_args_map`` like any other
args-shaped layer.
"""

import ast
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request

from app.arbiter import harvest_catalog_pair
from app.argline import normalize_args_map, parse_argline, render_argline
from app.compose_import import import_compose
from app.engines import EngineError
from app.events import log_event
from app.facts import resolve_facts
from app.ladder import LAYERS, resolve_settings
from app.observe import spark_node_id
from app.settings_store import KINDS
from app.validate_settings import validate_settings

router = APIRouter(tags=["settings"])


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@router.get("/settings/catalog/{node}/{engine}")
def get_catalog(node: str, engine: str, request: Request):
    deck = request.app.state.deck
    entry = deck["characteristics_store"].entry(f"engine/{node}/{engine}")
    catalog = entry.get("option_catalog")
    if not catalog:
        return None
    # harvested_ts: the field's own derived_ts. Additive — validate_settings
    # reads only .get("options") (app/validate_settings.py:81).
    return {**catalog["value"], "harvested_ts": catalog.get("derived_ts")}


@router.post("/settings/harvest/{node}/{engine}")
def harvest_now(node: str, engine: str, request: Request) -> dict:
    """Manual catalog refresh — the All-options Refresh button. force=True:
    a human asking again is exactly the case the version gate must not
    swallow (a broken auto-harvest looks 'current' forever otherwise).

    Uses ``log_event`` directly rather than the Watcher's own ``self._log``
    dedup wrapper: a human pressing this button is not a spam loop the way
    a stuck per-tick condition is, so every press should land its own event
    line, not get silently swallowed as a repeat of the last one.
    """
    deck = request.app.state.deck
    pairs = [tuple(p) for p in deck.get("configurable_engines") or []]
    if (node, engine) not in pairs:
        raise ValueError(
            f"({node}, {engine}) is not a configurable pair; configurable: {pairs}")
    if deck.get("engine_exec") is None:
        raise HTTPException(status_code=503, detail="no engine exec wired")
    return harvest_catalog_pair(
        deck["engine_exec"], deck["characteristics_store"],
        lambda kind, detail: log_event(deck["events_path"], kind, detail),
        node, engine, now=_now_iso(), force=True)


@router.get("/settings/effective/{node}/{engine}/{model:path}")
def get_effective(
    node: str, engine: str, model: str, request: Request, layers: str | None = None
) -> dict:
    deck = request.app.state.deck
    resolved = _resolve(deck, node, engine, model)
    catalog = get_catalog(node, engine, request)
    facts = _facts_for(deck, model)

    # Design decision 3 (Plan C2, Task 7): the argline — and, by extension,
    # anything ever SHIPPED to an engine (app.routers.spark.spark_reload) —
    # renders DECLARED layers (engine/model/engine_model) only. The two
    # DERIVED layers (engine_defaults, checkpoint_recommendations) are the
    # engine's own applied behavior, never re-asserted back at it as flags;
    # 'resolved' below still carries them, full provenance intact, for the
    # UI's structured view.
    declared = _declared_only(resolved)
    argline = render_argline({k: v["value"] for k, v in declared.items()})

    view = resolved
    if layers is not None:
        names = [name for name in layers.split(",") if name]
        unknown = sorted(set(names) - set(LAYERS))
        if unknown:
            raise ValueError(f"unknown layer(s) {unknown}; expected one of {LAYERS}")
        # A display filter on the structured view only — warnings and the
        # argline above are computed from the FULL resolution regardless,
        # so this param can never silently change what would actually ship.
        view = {k: v for k, v in resolved.items() if v["layer"] in names}

    return {
        "resolved": view,
        "argline": argline,
        "warnings": validate_settings(resolved, catalog, facts),
    }


@router.post("/settings/preview")
def preview(body: dict, request: Request) -> dict:
    """Parse typed text or render args map without saving — the text field's live feedback."""
    argline_in, args_in = body.get("argline"), body.get("args")
    if (argline_in is None) == (args_in is None):
        raise ValueError("preview requires exactly one of 'argline' or 'args'")
    if args_in is not None:
        parsed = normalize_args_map(args_in)
    else:
        parsed = parse_argline(argline_in)
    argline = render_argline(parsed)
    resolved = {k: {"value": v, "origin": "declared", "layer": "engine_model"}
                for k, v in parsed.items()}
    catalog = None
    node, engine = body.get("node"), body.get("engine")
    if node and engine:
        catalog = get_catalog(node, engine, request)
    facts = _facts_for(request.app.state.deck, body.get("model", ""))
    return {"parsed": parsed, "argline": argline, "warnings": validate_settings(resolved, catalog, facts)}


@router.post("/settings/adopt/{node}/{engine}")
def adopt(node: str, engine: str, request: Request) -> dict:
    """Sweep a node's real compose profiles into the settings store — the
    "adopt" half of adopt-then-own (Plan C2). Only ``(spark_node_id(),
    "vllm")`` is adoptable in C2 (an unknown pair is a ValueError -> 422,
    same family as this router's other save-time rejections); every other
    profile on the node (ds4, comfyui, ...) is reported 'skipped' rather
    than guessed at.

    Never clobbers: a scope whose args namespace already has something in
    it is left untouched and reported 'kept', not overwritten — adopt runs
    once against a live node, and re-running it must not stomp an edit a
    human already made.

    Also records the profile -> identity map (which compose service and
    container each profile is) as a characteristics field, so Tasks 6+7
    (drift translation, reload) can go from a profile name to the
    engine_models scope key without re-parsing compose themselves.

    Re-adopt is ADDITIVE (final branch review, 2026-08-07):
    ``put_fields`` REPLACES the field it is handed, so building
    ``identities`` from the current sweep alone evicted every profile whose
    fetch/parse happened to fail this time — a transient node hiccup, and
    that profile's drift silently reverts to verbatim scopes (drift goes
    dark, the failure Decision 9 exists to prevent) and its reload 409s
    "has no adopted identity". The map is therefore merged over the
    previous one: a profile that IS still on the node but was unreadable
    keeps its old entry. Nothing else is carried forward — a profile the
    sweep read successfully is authoritative about itself, and one absent
    from ``status()`` is gone from the node.

    Per-profile isolation (review round 1 fix, 2026-08-07): fetching
    (``SparkClient.get_compose``, ``EngineError`` on transport/HTTP failure)
    and parsing (``import_compose``, ``ValueError`` on malformed YAML /
    multi-service / non-list ``command:``) run PER PROFILE inside the loop,
    not around the whole route. One bad profile must not fail the entire
    request with a bare 422/502 after earlier profiles' ``store.put()``
    calls have already committed — that would leave scopes written that the
    identity map below never records, with no partial-success indication
    to the caller at all. A profile whose fetch or parse fails is reported
    under ``skipped`` with a reason naming the exception, exactly like the
    "no /model mount" and non-vllm skip reasons, and the sweep continues.
    The route-level guards (unknown node/engine -> 422, no spark client ->
    503, ``spark.status()`` itself failing) are unaffected — those are
    whole-request preconditions, not per-profile outcomes.
    """
    deck = request.app.state.deck
    if (node, engine) != (spark_node_id(), "vllm"):
        raise ValueError(f"only {spark_node_id()}/vllm is adoptable in C2")
    spark = deck.get("spark")
    if spark is None:
        raise HTTPException(status_code=503, detail="spark engine is not configured")

    store = deck["settings_store"]
    characteristics = deck["characteristics_store"]
    previous = (characteristics.entry(f"engine/{node}/{engine}")
                .get("profile_identities") or {}).get("value") or {}
    adopted, kept, skipped, identities = [], [], [], {}
    unreadable: set[str] = set()
    for meta in spark.status()["profiles"]:
        if meta["engine"] != "vllm":
            skipped.append({"profile": meta["name"], "engine": meta["engine"]})
            continue
        try:
            imported = import_compose(spark.get_compose(meta["name"]))
        except (ValueError, EngineError) as exc:
            # One unreadable/malformed profile must not fail the whole
            # sweep (see this function's docstring, "Per-profile
            # isolation") — report it and keep going.
            skipped.append({"profile": meta["name"], "engine": "vllm",
                            "reason": f"{type(exc).__name__}: {exc}"})
            unreadable.add(meta["name"])
            continue
        identity = imported["identity"]
        if identity is None:
            skipped.append({"profile": meta["name"], "engine": "vllm",
                            "reason": "no /model mount"})
            continue
        identities[meta["name"]] = {"identity": identity,
                                    "service": imported["service"],
                                    "container_name": imported["container_name"]}
        key = f"{node}/{engine}|{identity}"
        if store.scope("engine_models", key).get("args"):
            kept.append(key)
            continue
        note = f"adopted from compose-{meta['name']}.yaml"
        store.put("engine_models", key, "args", imported["args"],
                  note=(imported["notes"].get("args") or note))
        if imported["env"]:
            store.put("engine_models", key, "env", imported["env"], note=note)
        if imported["container"]:
            store.put("engine_models", key, "container", imported["container"], note=note)
        adopted.append(key)

    # Merge, never replace — see this function's docstring, "Re-adopt is
    # additive". `unreadable` is the ONLY carry-forward set: a profile this
    # sweep read successfully is authoritative about itself (dropped its
    # /model mount, stopped being vllm), and a profile absent from status()
    # is gone from the node.
    merged = {name: previous[name] for name in unreadable if name in previous}
    merged.update(identities)
    if merged or previous:
        characteristics.put_fields(
            f"engine/{node}/{engine}",
            {"profile_identities": {"value": merged,
                                    "source": "compose import",
                                    "derived_ts": _now_iso()}})
    return {"adopted": adopted, "kept": kept, "skipped": skipped}


@router.get("/settings/{kind}/{key:path}")
def get_settings(kind: str, key: str, request: Request) -> dict:
    # Symmetry with PUT (optional fix, final branch review, 2026-08-07):
    # SettingsStore.put() rejects an unknown `kind` with a ValueError -> 422
    # (see module docstring), but scope()/GET had no equivalent check and
    # silently returned 200 {} for a typo'd kind -- indistinguishable from
    # a real, empty scope of a valid kind.
    if kind not in KINDS:
        raise ValueError(f"unknown scope kind {kind!r}; expected one of {KINDS}")
    return _public_scope(request.app.state.deck, kind, key)


@router.put("/settings/{kind}/{key:path}")
def put_settings(kind: str, key: str, body: dict, request: Request) -> dict:
    deck = request.app.state.deck
    namespace, values = body.get("namespace"), body.get("values")
    if namespace is None or values is None:
        raise ValueError("PUT /settings requires both 'namespace' and 'values'")
    deck["settings_store"].put(kind, key, namespace, values, note=body.get("note"), remove=body.get("remove"))
    return _public_scope(deck, kind, key)


def _public_scope(deck: dict, kind: str, key: str) -> dict:
    """``scope()`` minus ``updated_ts`` and ``journal`` — internal write-tracking
    bookkeeping (see module docstring), never part of this response's contract.
    A fresh dict: ``scope()`` already returns a freshly-loaded object (not a
    live store reference), but popping from a copy here keeps that
    non-aliasing an explicit property of THIS function too, not an
    accident inherited from the store."""
    scope = dict(deck["settings_store"].scope(kind, key))
    scope.pop("updated_ts", None)
    scope.pop("journal", None)
    return scope


# Sentinel: this decoded default's honest rendering is an absent flag, not
# a value — see _decode_harvested_default and this module's docstring,
# "Engine-default decoding".
_DROP = object()


def _decode_harvested_default(raw):
    """Decode one harvested option's ``repr()``'d default (see
    app.harvest module docstring: PROBE_SOURCE stores ``repr(action.
    default)``, the cache's raw truth, unconditionally) back to the Python
    value it represents, or ``_DROP`` when that value's honest rendering is
    an absent flag — see this module's docstring, "Engine-default
    decoding". ``ast.literal_eval`` failing (a string that never was a
    Python repr, or a repr shape it doesn't cover) falls back to the raw
    string: a repr this module can't decode is still a string value, and
    dropping it here would lose real information a scalar-string default
    would keep.
    """
    try:
        decoded = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        decoded = raw
    if decoded is False or decoded is None or decoded in ([], {}):
        return _DROP
    return decoded


def _resolve(deck: dict, node: str, engine: str, model: str) -> dict:
    store = deck["settings_store"]
    characteristics = deck["characteristics_store"]

    engine_key = f"{node}/{engine}"
    catalog_entry = characteristics.entry(f"engine/{engine_key}").get("option_catalog")
    engine_defaults = {}
    if catalog_entry:
        for name, option in catalog_entry["value"]["options"].items():
            raw_default = option.get("default")
            if raw_default in (None, "None"):
                continue
            decoded = _decode_harvested_default(raw_default)
            if decoded is not _DROP:
                engine_defaults[name] = decoded

    recommended = characteristics.entry(f"model/{model}").get("recommended_sampling")

    # AMENDED 2026-08-07 (Task 3 review finding): the two DERIVED layers are
    # assembled outside SettingsStore.put(), the only place the ruled
    # normalization axes are otherwise enforced — generation_config.json
    # sampling values arrive as raw ints/floats. Every layer entering
    # resolve_settings must pass through argline.normalize_args_map (the
    # canonical home of the axes, hoisted there in Task 3's fix round), or
    # resolved output leaks int-where-str shapes downstream code was told
    # are impossible. Store layers are already normalized on write.
    return resolve_settings(
        engine_defaults=normalize_args_map(engine_defaults),
        checkpoint_recommendations=normalize_args_map((recommended or {}).get("value", {})),
        engine=store.scope("engines", engine_key).get("args", {}),
        model=store.scope("models", model).get("args", {}),
        engine_model=store.scope("engine_models", f"{engine_key}|{model}").get("args", {}),
    )


def _declared_only(resolved: dict) -> dict:
    """Design decision 3 (Plan C2, Task 7): the DECLARED layers of a
    resolution — engine/model/engine_model, never engine_defaults or
    checkpoint_recommendations. Shared by get_effective's argline and
    app.routers.spark.spark_reload's shipped document, so the two can never
    drift apart on what "declared-only" means."""
    return {k: v for k, v in resolved.items() if v["origin"] == "declared"}


def _resolve_env(deck: dict, node: str, engine: str, model: str) -> dict:
    """Env's most-specific-wins resolution — no derived layers exist for
    env (engine_defaults/checkpoint_recommendations are args-only concepts:
    a harvested CLI flag's default, a checkpoint's generation_config.json
    sampling), so this reuses app.ladder.resolve_settings with both derived
    layers empty rather than re-implementing "most specific wins" a second
    time for env."""
    store = deck["settings_store"]
    engine_key = f"{node}/{engine}"
    resolved = resolve_settings(
        engine_defaults={},
        checkpoint_recommendations={},
        engine=store.scope("engines", engine_key).get("env", {}),
        model=store.scope("models", model).get("env", {}),
        engine_model=store.scope("engine_models", f"{engine_key}|{model}").get("env", {}),
    )
    return {key: entry["value"] for key, entry in resolved.items()}


def _facts_for(deck: dict, model: str) -> dict:
    if not model:
        return {}
    return resolve_facts(
        deck["characteristics_store"].entry(f"model/{model}"),
        deck["declared_store"].entry(f"model/{model}"),
    )
