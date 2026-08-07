"""Settings resolution — five layers, per key, most specific wins.

    engine_defaults           (derived: harvested from the engine)
    checkpoint_recommendations (derived: generation_config.json)
    engine                    (declared: <node>/<engine>)
    model                     (declared: identity, node-neutral)
    engine_model              (declared: the most specific)

**Merge is per KEY, never per blob.** Overriding one flag at the most
specific scope must leave every other flag from lower layers intact.
Whole-blob replacement is how config systems rot: you set one thing and
silently lose four.

**``None`` at a higher layer unsets a lower one.** Dropping an inherited
flag needs a representation, and "absent" is a real, correct configuration:
heretic's 2026-07-31 fix was the ABSENCE of ``--quantization``, letting vLLM
auto-detect the compressed-tensors checkpoint. A ladder that could only add
flags could not express the fix.

Provenance is retained per key (which layer, derived or declared) so the UI
can show where a value came from and offer "revert to inherited".

No normalization here, by decision. app.argline documents two
RULING-2026-08-07 normalization axes (singleton list -> scalar, numeric ->
string) and app.settings_store applies both to ``args`` values on write, so
layers sourced from the store already arrive normalized. A layer sourced
elsewhere -- code-baked defaults, or a live line run through
``parse_argline`` -- is not guaranteed to be. This module never compares a
lower layer's value against a higher one to decide a winner; presence of a
key at a more specific layer is enough, and that key's value replaces the
lower layer's *verbatim*, unexamined. So a mixed int/str or scalar/list
representation of "the same" value across layers cannot defeat
most-specific-wins or produce spurious provenance -- there is no equality
check for it to defeat. Normalizing here would be pure overhead for a
comparison this module never performs; it stays the responsibility of
whichever layer produces the value (store on write, or the caller for
code-baked/live-harvested layers that skip the store).

This module receives already-scoped layer dicts, not raw store contents --
it never constructs an ``engine_models`` compound key (``<node>/<engine>|
<model>``) itself, so the "|" separator collision (an engine name
containing "|") that key convention warns about never arises here; it is
entirely the caller's concern when it calls ``settings_store.scope()``.

Pure functions, no I/O.
"""

LAYERS = ("engine_defaults", "checkpoint_recommendations", "engine", "model", "engine_model")

_DERIVED_LAYERS = ("engine_defaults", "checkpoint_recommendations")


def resolve_settings(
    *,
    engine_defaults: dict,
    checkpoint_recommendations: dict,
    engine: dict,
    model: dict,
    engine_model: dict,
) -> dict[str, dict]:
    """Return ``{key: {value, origin, layer}}`` for the effective settings."""
    by_layer = {
        "engine_defaults": engine_defaults,
        "checkpoint_recommendations": checkpoint_recommendations,
        "engine": engine,
        "model": model,
        "engine_model": engine_model,
    }

    resolved: dict[str, dict] = {}
    for layer in LAYERS:
        for key, value in by_layer[layer].items():
            if value is None:
                # Explicit unset — remove anything a lower layer contributed.
                resolved.pop(key, None)
                continue
            resolved[key] = {
                "value": value,
                "origin": "derived" if layer in _DERIVED_LAYERS else "declared",
                "layer": layer,
            }
    return resolved
