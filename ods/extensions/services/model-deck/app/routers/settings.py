"""Settings router — the two surfaces over one store.

``GET/PUT /settings/{kind}/{key}`` is the store. ``GET
/settings/effective/...`` resolves the ladder and renders the SAME map as a
command line, so the chip panel and the text field can never disagree —
they are literally the same data rendered twice. ``POST /settings/preview``
parses typed text without saving, which is what makes the text field feel
live.

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

from app.argline import normalize_args_map, parse_argline, render_argline
from app.compose_import import import_compose
from app.engines import EngineError
from app.facts import resolve_facts
from app.ladder import resolve_settings
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
    return catalog["value"] if catalog else None


@router.get("/settings/effective/{node}/{engine}/{model:path}")
def get_effective(node: str, engine: str, model: str, request: Request) -> dict:
    deck = request.app.state.deck
    resolved = _resolve(deck, node, engine, model)
    catalog = get_catalog(node, engine, request)
    facts = _facts_for(deck, model)

    return {
        "resolved": resolved,
        "argline": render_argline({k: v["value"] for k, v in resolved.items()}),
        "warnings": validate_settings(resolved, catalog, facts),
    }


@router.post("/settings/preview")
def preview(body: dict, request: Request) -> dict:
    """Parse typed text without saving — the text field's live feedback."""
    parsed = parse_argline(body.get("argline", ""))
    resolved = {k: {"value": v, "origin": "declared", "layer": "engine_model"}
                for k, v in parsed.items()}
    catalog = None
    node, engine = body.get("node"), body.get("engine")
    if node and engine:
        catalog = get_catalog(node, engine, request)
    facts = _facts_for(request.app.state.deck, body.get("model", ""))
    return {"parsed": parsed, "warnings": validate_settings(resolved, catalog, facts)}


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
    adopted, kept, skipped, identities = [], [], [], {}
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

    if identities:
        deck["characteristics_store"].put_fields(
            f"engine/{node}/{engine}",
            {"profile_identities": {"value": identities,
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
    deck["settings_store"].put(kind, key, namespace, values, note=body.get("note"))
    return _public_scope(deck, kind, key)


def _public_scope(deck: dict, kind: str, key: str) -> dict:
    """``scope()`` minus ``updated_ts`` — internal write-tracking bookkeeping
    (see module docstring), never part of this response's contract. A
    fresh dict: ``scope()`` already returns a freshly-loaded object (not a
    live store reference), but popping from a copy here keeps that
    non-aliasing an explicit property of THIS function too, not an
    accident inherited from the store."""
    scope = dict(deck["settings_store"].scope(kind, key))
    scope.pop("updated_ts", None)
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


def _facts_for(deck: dict, model: str) -> dict:
    if not model:
        return {}
    return resolve_facts(
        deck["characteristics_store"].entry(f"model/{model}"),
        deck["declared_store"].entry(f"model/{model}"),
    )
