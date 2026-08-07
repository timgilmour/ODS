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

A one-element list (``--served-model-name`` takes a single value in some
profiles, six in others) is lexically identical to a scalar once rendered —
``--flag v`` gives a parser no signal that ``v`` came from a ``list`` rather
than a ``str``. Losing that distinction would silently change the value's
type on the next edit, so a singleton list is rendered with one extra
trailing empty-string token (``shlex.quote("")`` == ``''``) that scalars
never emit, and parsing strips it back off. The accepted, narrow trade-off:
a flag typed by a human with a genuine trailing empty argument (``--tag foo
''`` meaning two values, the second blank) parses the same way. No case in
this module's tests, or in a live sparky profile, does that.

Positional tokens (``serve /model`` leads every vLLM command array) are
preserved under the reserved key ``_positional``. Dropping them would
silently change what gets launched.

Unknown flags are preserved verbatim: this module has no notion of which
flags are legal. Validation happens elsewhere (app.validate_settings) and
warns rather than blocking, so a flag a new engine version added is always
reachable.

A parse never raises. A human is typing into this field; unbalanced quotes
degrade to a best-effort parse rather than an exception.
"""

import shlex

POSITIONAL_KEY = "_positional"


def render_argline(settings: dict) -> str:
    """Render a settings map as a shell-safe command line."""
    parts: list[str] = []

    for token in settings.get(POSITIONAL_KEY, []):
        parts.append(shlex.quote(str(token)))

    for key, value in settings.items():
        if key == POSITIONAL_KEY:
            continue
        flag = f"-{key}" if len(key) == 1 else f"--{key}"
        if value is True:
            parts.append(flag)
        elif isinstance(value, list):
            parts.append(flag)
            parts.extend(shlex.quote(str(v)) for v in value)
            if len(value) == 1:
                # Disambiguate from a scalar — see module docstring.
                parts.append(shlex.quote(""))
        else:
            parts.append(flag)
            parts.append(shlex.quote(str(value)))

    return " ".join(parts)


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
        elif len(values) == 2 and values[1] == "":
            # render_argline's singleton-list marker — see module docstring.
            _assign_many(settings, current, values[:1])
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
