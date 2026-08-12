"""One Prometheus-exposition line-summer shared by the engine clients.

Both LemonadeClient.activity() (llama.cpp token counters, the idle-release
signal) and SparkClient.busy_requests() (vLLM/ds4 in-flight gauges, the
swap busy-guard) sum the values of exposition lines whose metric name
starts with a caller-supplied prefix tuple. They used to carry separate
hand-rolled parsers with quietly different error posture — a format edge
fixed in one (comment lines, non-finite values, trailing timestamp tokens)
silently stayed broken in the other (max-review c22).

The parsing lives here once and is deliberately STRICT: any matching line
whose value is missing, unparseable, or non-finite raises ValueError naming
the line. The POSTURE stays with each caller — lemonade maps any ValueError
to None (activity is best-effort), spark maps it to EngineError (an
unreadable busy signal must never read as idle).
"""

import math


def sum_matching(text: str, prefixes: tuple[str, ...]) -> tuple[float, bool]:
    """Sum values of exposition lines whose metric name starts with one of
    ``prefixes``. Returns ``(total, matched)`` where ``matched`` is True if
    any line's metric name matched, even a zero-valued one.

    Raises ValueError for a matching line with a missing, unparseable, or
    non-finite value (``float('NaN')``/``float('1e999')`` parse fine but
    ``int(total)`` downstream would crash outside the caller's error
    vocabulary). Blank and comment lines are skipped; a line with a
    trailing exposition timestamp is refused rather than silently summing
    the wrong token.
    """
    total = 0.0
    matched = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if not parts[0].startswith(prefixes):
            continue
        matched = True
        if len(parts) != 2:
            raise ValueError(f"unparseable metric line: {raw!r}")
        try:
            value = float(parts[1])
        except ValueError:
            raise ValueError(f"unparseable metric line: {raw!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric value: {raw!r}")
        total += value
    return total, matched
