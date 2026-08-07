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

``GET``/``PUT /settings/{kind}/{key}`` strip ``updated_ts`` out of the
returned scope before it reaches a caller: it is write-tracking bookkeeping
for ``app.routers._settings_drift``'s comparison, stamped by
``SettingsStore.put``, not a documented field of this response shape — and
leaking it would make it part of the API's contract by accident.
"""

from fastapi import APIRouter, Request

from app.argline import normalize_args_map, parse_argline, render_argline
from app.facts import resolve_facts
from app.ladder import resolve_settings
from app.validate_settings import validate_settings

router = APIRouter(tags=["settings"])


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


@router.get("/settings/{kind}/{key:path}")
def get_settings(kind: str, key: str, request: Request) -> dict:
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


def _resolve(deck: dict, node: str, engine: str, model: str) -> dict:
    store = deck["settings_store"]
    characteristics = deck["characteristics_store"]

    engine_key = f"{node}/{engine}"
    catalog_entry = characteristics.entry(f"engine/{engine_key}").get("option_catalog")
    engine_defaults = {}
    if catalog_entry:
        engine_defaults = {
            name: option["default"]
            for name, option in catalog_entry["value"]["options"].items()
            if option.get("default") not in (None, "None")
        }

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
