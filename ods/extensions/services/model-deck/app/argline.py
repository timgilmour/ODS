"""Settings map <-> command line, losslessly.

The chip panel and the free-text field are two views of ONE store. This
module is the hinge between them, which makes the round trip the integrity
guarantee of the whole settings feature: chips -> text -> chips must be the
identity, or a human's edit gets silently altered.

Value shapes, all of which appear in a live sparky profile:

* ``True``        — bare flag (``--enable-chunked-prefill``)
* ``str``/``int`` — scalar (``--max-model-len 262144``)
* ``list[str]``   — multi-value (``--served-model-name`` takes SIX values on
  mm27b; rendering it once per value would be a different command line)

A JSON blob (``--speculative-config '{"method":"dflash",...}'``) is just a
scalar whose text contains spaces and quotes — it survives because
rendering goes through shlex.quote and parsing through shlex.split.

A one-element list and a scalar render IDENTICALLY — ``render({"x": ["v"]})
== render({"x": "v"})`` — and a single trailing token always parses to a
scalar. This is a deliberate normalization, not a gap: ``--flag v`` gives a
parser no signal that ``v`` came from a ``list`` rather than a ``str``, and
the engine draws no distinction either (one ``--served-model-name`` value is
one argument, list or not). RULING 2026-08-07: an earlier draft disambiguated
with an invented trailing empty-string token. Rejected — this text is also a
real engine command line, and a token a human never typed (an empty-string
CLI argument) is as much a correctness bug as dropping one would be. So the
MAP-level round trip is exact modulo this one normalization (a singleton
list collapses to its scalar through text); the TEXT-level round trip stays
exact, byte for byte, for every line this module can produce — no invented
tokens, ever.

Normalization axis #2, RULING 2026-08-07: ``parse_argline`` always returns
``str`` (or ``True``) — text cannot encode ``int`` vs ``str`` any more than
it can encode list-of-one vs scalar, and for the same reason: ``--flag 5``
has one spelling regardless of which Python type produced it.
``render_argline`` still accepts ``int``/``float`` scalars and stringifies
them, so a caller need not pre-convert. Net effect: the map round trip is
exact modulo TWO normalizations (singleton list -> scalar, numeric ->
string). Callers that store or diff resolved settings (app.ladder, Task 2
onward) must normalize on write, or compare via rendered text — not via
``==`` on the raw map.

``normalize_args_map`` (below) is THE canonical enforcement of both axes,
plus the empty-list-carries-no-opinion rule (RULING 2026-08-07 review) —
hoisted here from app.settings_store (review, 2026-08-07) after a review
finding on app.ladder (Task 3): resolved ladder output could carry a raw
int (e.g. ``checkpoint_recommendations`` built straight from
generation_config.json sampling values) — a shape Tasks 1/2 ruled
impossible — because normalization lived ONLY inside
``SettingsStore.put()``, and a derived layer (code-baked engine defaults, a
live-harvested line, checkpoint recommendations) never passes through the
store at all. Any args-shaped layer assembled OUTSIDE the store must be
passed through ``normalize_args_map`` before it enters
``app.ladder.resolve_settings`` — the ladder is deliberately value-blind
(see app.ladder module docstring) and will resolve and serve whatever
shape it is handed, normalized or not.

``POSITIONAL_KEY`` is EXEMPT from the singleton-list axis — CRITICAL, fixed
2026-08-07 (final branch review, finding F1). The singleton-collapse axis
exists because, for a NAMED flag, ``render_argline``/``parse_argline``
cannot round-trip list-of-one vs scalar through space-separated text at
all — ``--flag v`` has one spelling regardless of which shape produced it.
A positional token carries no such ambiguity: it is always one element of
an ordered sequence, whether that sequence has length one or six.
Collapsing it anyway silently turns a ``list[str]`` into a bare ``str``,
and ``render_argline`` iterates a bare string character by character —
``parse_argline("serve --max-model-len 100")`` through
``normalize_args_map`` used to come back as ``{"_positional": "serve",
...}``, which rendered as ``'s e r v e --max-model-len 100'``. Persisted
corruption of typed input, the plan's one unacceptable failure mode.
``normalize_args_map`` keeps ``POSITIONAL_KEY``'s value ``list[str]``
shaped always — one element or many — while still applying the numeric ->
string axis per element and the empty-list-drop rule to the list as a
whole.

Dash-shaped values — CRITICAL, fixed 2026-08-07: a value that itself starts
with ``-`` (``--stop-token --foo``, or a ``--served-model-name`` list with
``-basemodel`` as one element) is, once space-separated, indistinguishable
from the start of a NEW flag — the parser has only whitespace to go on and
cannot know ``--foo`` is stop-token's value rather than a bare flag of its
own. Left unfixed this silently reassigns real values to bogus keys (the
one unacceptable failure mode). The fix lives entirely on the RENDER side —
the renderer knows what is a value; the parser fundamentally cannot: a
dash-shaped scalar renders in equals-form (``--stop-token=--foo``, one
token, no whitespace boundary to misread). For a list the switch is per
LIST, not per element: if ANY element is dash-shaped, the WHOLE list
renders as repeated equals-form occurrences (``--served-model-name=a
--served-model-name=-basemodel --served-model-name=c``) instead of mixing
space-form runs with equals-form islands. Mixing is provably correct too,
but only if a fresh bare ``--flag`` occurrence reopens after every
equals-form island — parsing an equals-form token resets the parser's
"collecting values for this flag" state, so a bare value straight after one
would misparse as positional. That is meaningfully more logic to prove for
a shape with no evidence in any live profile. A list with no dash-shaped
element keeps the plain single-occurrence, space-joined form unchanged.

A negative number (``-5``) is not dash-shaped for this purpose — see
``_is_negative_number``, shared by parse and render so the two sides can
never disagree about what counts as a flag boundary.

Positional tokens (``serve /model`` leads every vLLM command array) are
preserved under the reserved key ``_positional``. Dropping them would
silently change what gets launched.

Unknown flags are preserved verbatim: this module has no notion of which
flags are legal. Validation happens elsewhere (app.validate_settings) and
warns rather than blocking, so a flag a new engine version added is always
reachable. The same schema-lessness has a sharp edge worth flagging for
Task 2+ authors: a bare flag followed by an unrelated stray token reads as
that flag's VALUE, not as a boolean plus something else unexplained — this
parser has no notion of flag arity either, and giving it one would need a
catalog, which would violate the same purity.

A parse never raises. A human is typing into this field; unbalanced quotes
degrade to a best-effort parse rather than an exception.
"""

import shlex
import warnings

POSITIONAL_KEY = "_positional"

# Sentinel: this args value carries no argline opinion and must be dropped
# from the map, not returned — see _normalize_args_value and
# normalize_args_map. Hoisted from app.settings_store, review 2026-08-07.
_DROP = object()


def _require_renderable(key: str, value) -> None:
    """Refuse anything that has no honest argline rendering [max-review c49].

    Applied at EVERY place a value becomes a token — bare values, list
    ELEMENTS, and positional tokens — because the fall-through `str(value)`
    is reached from all three. The first version of this guard sat only on
    the bare-value branch, below the list branch and after the positional
    loop, so `{"served-model-name": ["a", {"a": 1}]}` and
    `{POSITIONAL_KEY: ["serve", {"a": 1}]}` both still rendered a Python
    repr into a launch argline — c49 verbatim, through the very boundary
    whose comment claims to cover hand-edited files.

    ``None`` never reaches this function: every caller skips it first. It is
    a load-bearing value meaning EXPLICIT UNSET (app/ladder.py:73 pops the
    key and everything a lower layer contributed), so its honest rendering is
    nothing at all. Refusing it instead made a scope containing one
    permanently un-viewable and un-editable through the deck — a 422 the
    operator had no way to clear.

    That skip is at THREE sites, not one. An earlier version put it only on
    the bare-value branch, which left list elements and positional tokens
    refusing None and reproduced the same wedge one level down — this time
    also 422ing POST /api/spark/reload, since the ladder pops only TOP-LEVEL
    Nones and a list containing one survives resolution.
    """
    if not isinstance(value, (str, int, float)):
        raise ValueError(
            f"setting {key!r} has an unrenderable value: {value!r}")


def _argv_tokens(settings: dict) -> list[str]:
    """The dispatch shared by render_argv and render_argline: positionals,
    then one flag per remaining key, UNQUOTED. Quoting is each caller's own
    concern -- render_argv ships tokens verbatim into a compose `command:`
    array, render_argline joins them for a shell."""
    parts: list[str] = []

    for token in settings.get(POSITIONAL_KEY) or []:
        if token is None:
            continue          # see the None note in _require_renderable
        _require_renderable(POSITIONAL_KEY, token)
        parts.append(str(token))

    for key, value in settings.items():
        if key == POSITIONAL_KEY:
            continue
        flag = f"-{key}" if len(key) == 1 else f"--{key}"
        if value is None:
            # EXPLICIT UNSET, not a missing value: app/ladder.py:73 pops the
            # key and anything a lower layer contributed, so by the time a
            # RESOLVED map is rendered there is nothing left to emit. Raw
            # declared layers are rendered too (the resolved-settings view),
            # and there the honest rendering of "unset" is likewise nothing.
            continue
        if value is True:
            parts.append(flag)
        elif isinstance(value, list):
            # Drop None ELEMENTS the same way the bare-value branch drops a
            # bare None: nothing is the honest rendering of "no value", and
            # refusing here wedged real ship paths — a persisted [None] made
            # GET effective, the preview AND POST /api/spark/reload all 422,
            # i.e. the endpoints that would show the operator what to fix
            # were the ones failing. The wire refuses NEW list-Nones
            # (_normalize_args_value); this keeps already-persisted ones
            # renderable so they can be edited out.
            value = [v for v in value if v is not None]
            for element in value:
                _require_renderable(key, element)
            if not value:
                continue      # a list of nothing but Nones says nothing
            if any(_looks_like_a_flag(v) for v in value):
                # A dash-shaped element anywhere forces the whole list to
                # equals-form -- see module docstring.
                parts.extend(f"{flag}={v}" for v in value)
            else:
                parts.append(flag)
                parts.extend(str(v) for v in value)
        elif _looks_like_a_flag(value):
            parts.append(f"{flag}={value}")
        else:
            # The SECOND of two boundaries: normalize_args_map refuses these
            # at the wire (PUT /api/settings -> 422). That gate cannot cover
            # this one — this renderer also consumes settings read back from
            # a settings.json that was hand-edited, or written before the
            # gate existed. bool is excluded by the `is True` branch above.
            _require_renderable(key, value)
            parts.append(flag)
            parts.append(str(value))

    return parts


def render_argv(settings: dict) -> list[str]:
    """The settings document's argv: same dispatch as render_argline, no
    shell quoting. Compose `command:` arrays consume tokens verbatim, so
    quoting here would ship literal quote characters into the process."""
    return _argv_tokens(settings)


def render_argline(settings: dict) -> str:
    """Render a settings map as a shell-safe command line."""
    return " ".join(shlex.quote(token) for token in _argv_tokens(settings))


def parse_argline(text: str) -> dict:
    """Parse a command line back into a settings map.

    Never raises: unbalanced quotes fall back to a whitespace split so the
    operator's text is preserved rather than rejected.
    """
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()

    settings: dict = {}
    positional: list[str] = []
    current: str | None = None
    values: list[str] = []

    def flush() -> None:
        nonlocal current, values
        if current is None:
            return
        if not values:
            settings[current] = True
        elif len(values) == 1:
            _assign(settings, current, values[0])
        else:
            _assign_many(settings, current, values)
        current, values = None, []

    for token in tokens:
        if token.startswith("-") and not _is_negative_number(token):
            flush()
            name = token.lstrip("-")
            if "=" in name:
                name, _, value = name.partition("=")
                _assign(settings, name, value)
                current = None
            else:
                current = name
        elif current is None:
            positional.append(token)
        else:
            values.append(token)

    flush()

    if positional:
        settings[POSITIONAL_KEY] = positional
    return settings


def _assign(settings: dict, key: str, value: str) -> None:
    """A repeated flag collapses into a list rather than overwriting —
    ``--tag a --tag b`` means both, not the last one."""
    if key in settings:
        _assign_many(settings, key, [value])
        return
    settings[key] = value


def _assign_many(settings: dict, key: str, values: list[str]) -> None:
    existing = settings.get(key)
    if existing is None or existing is True:
        settings[key] = list(values)
    elif isinstance(existing, list):
        settings[key] = existing + list(values)
    else:
        settings[key] = [existing, *values]


def _is_negative_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_a_flag(value) -> bool:
    """True for anything parse_argline's own flag-boundary test would read
    as a new flag rather than a value -- the render-side half of the
    dash-shaped-value fix (see module docstring). Shares the exact
    predicate parse uses so the two sides can never disagree."""
    text = str(value)
    return text.startswith("-") and not _is_negative_number(text)


def _normalize_args_scalar(value):
    """One of the two RULING 2026-08-07 axes: int/float becomes str. bool
    is an int subclass in Python, so it is excluded explicitly -- ``True``
    is the bare-flag sentinel, not a number.

    A ``dict`` is REFUSED (see below) -- it has no argline rendering, and
    accepting it here persisted a value that later rendered as a Python
    repr [max-review c49]. Anything else (str, None, ...) is outside the
    documented axes and passes through unchanged rather than guessed at;
    ``None`` in particular is meaningful, not absent -- app/ladder.py:73
    reads it as an explicit unset."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        # The WIRE boundary [max-review c49]. PUT /api/settings takes JSON,
        # so an object here is operator input, and there is no argline a
        # mapping could mean — refuse it at entry (ValueError -> 422) rather
        # than persisting it for _argv_tokens to trip over at launch time.
        # Deliberately narrow: `None` and other types still pass through
        # unchanged, per this function's documented "outside the axes"
        # posture. The renderer holds the disk-side boundary this cannot
        # reach (settings written before this gate, or hand-edited).
        raise ValueError(f"setting value must not be a mapping, got {value!r}")
    return value


def _normalize_args_value(value):
    """Normalize one args value to argline's post-round-trip shape, or
    return _DROP if the value carries no argline opinion at all.

    A list of one collapses to its (normalized) element -- the other
    RULING 2026-08-07 axis. A list of zero is _DROP: RULING 2026-08-07
    (review) -- it renders byte-identical to no value at all (see module
    docstring), so keeping it as a key would claim an opinion the caller
    never expressed.
    """
    if isinstance(value, list):
        if not value:
            return _DROP
        if any(v is None for v in value):
            # A bare None means "unset this key" (app/ladder.py:73); INSIDE a
            # list it means nothing at all, and persisting one used to reach
            # the renderer and 422 every path that renders. Refuse the new
            # write rather than guess which element was meant.
            raise ValueError("a list value must not contain null")
        if len(value) == 1:
            return _normalize_args_scalar(value[0])
        return [_normalize_args_scalar(v) for v in value]
    return _normalize_args_scalar(value)


def _normalize_positional_value(value):
    """POSITIONAL_KEY's normalization, exempt from the singleton-list ->
    scalar axis -- see module docstring, "POSITIONAL_KEY is EXEMPT...".
    Always returns a ``list[str]`` (or ``_DROP`` for an empty list): a
    positional token has no scalar/list ambiguity to collapse, unlike a
    named flag's value. The numeric -> string axis still applies per
    element; a non-list input is wrapped into a one-element list first, so
    a caller handing a bare positional value still gets the canonical
    shape back.
    """
    if value is None:
        # Same unset meaning as any other key's bare None. Wrapping it into
        # [None] (what the generic path below would do) invented a positional
        # token nobody typed, and that [None] then 422'd every renderer.
        return _DROP
    items = value if isinstance(value, list) else [value]
    if not items:
        return _DROP
    if any(v is None for v in items):
        raise ValueError("a positional list must not contain null")
    return [_normalize_args_scalar(v) for v in items]


def normalize_args_map(values: dict) -> dict:
    """Normalize an args-shaped map to argline's post-round-trip shape --
    THE canonical enforcement of both RULING 2026-08-07 axes (singleton
    list -> scalar, numeric -> string) plus the empty-list-drop rule (see
    module docstring), MINUS the singleton-list axis for ``POSITIONAL_KEY``
    (F1, 2026-08-07 review -- see module docstring). A key whose value is
    an empty list carries no argline opinion and is dropped from the
    result with a ``UserWarning``, not returned -- matching
    app.settings_store's pre-hoist posture byte for byte.

    Any caller assembling an args-shaped layer outside app.settings_store
    (code-baked defaults, a live-harvested line through parse_argline,
    checkpoint recommendations from generation_config.json) must call this
    before the layer reaches app.ladder.resolve_settings.
    """
    normalized = {}
    for key, value in values.items():
        # The scalar normalizer refuses mappings (c49) but has no key in
        # scope; an operator reading a 422 needs to know WHICH setting was
        # refused, so the key is attached here — the one place that knows it,
        # and one that covers the nested-in-a-list path too.
        try:
            if key == POSITIONAL_KEY:
                n = _normalize_positional_value(value)
            else:
                n = _normalize_args_value(value)
        except ValueError as exc:
            raise ValueError(f"setting {key!r}: {exc}") from exc
        if n is _DROP:
            warnings.warn(
                f"app.argline: empty list for args key {key!r} carries no "
                "argline value (RULING 2026-08-07 review); dropped, not returned",
                stacklevel=2,
            )
            continue
        normalized[key] = n
    return normalized
