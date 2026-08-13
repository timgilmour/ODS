"""Import a docker-compose service definition into settings — the "adopt"
half of adopt-then-own (Plan C2, Task 2).

The later adopt sweep feeds a remote node's REAL compose files (one service each)
through here to seed the settings store, so what this module returns is not
an internal convenience shape — ``identity``, ``service`` and
``container_name`` are the fields the reload route and drift translation are
built on downstream (Tasks 3+). Renaming them here breaks those callers.

``command:`` parsing deliberately reuses ``app.argline.parse_argline`` (C1)
rather than a second parser: positionals, bare flags, multi-value flags and
JSON-blob scalars (``--speculative-config '{"method":...}'``) all come free
from that module having already solved the shell-quoting/list problem. The
compose YAML list is rejoined into one shell-quoted line first — a compose
``command:`` array and a shell command line are the same tokens, just two
different encodings of them.

Comments are handled separately from the YAML parse because YAML parsers
discard them: ``yaml.safe_load`` never sees a ``#`` line, so the only way to
recover one (e.g. compose-heretic.yaml's note on why --quantization is
deliberately absent) is a second, text-level pass over the RAW string this
function was handed, not the parsed document.
"""

import re
import shlex
from pathlib import Path

import yaml

from app.argline import parse_argline
from app.settings_store import CONTAINER_ALLOWLIST

# The vLLM convention this fleet standardizes on: the checkpoint directory
# is mounted read-only at this container path. Any other mount target is
# not a model mount (ds4 uses /models, comfyui mounts none) — identity is
# None rather than guessed at from an unrelated path.
_MODEL_MOUNT_TARGET = "/model"

# Matches a bare `command:` block-list header (optionally indented), never
# an inline `command: [...]` form — a comment scan only makes sense for the
# block form, since an inline flow sequence has nowhere to hide a comment
# that YAML would keep off the line.
_COMMAND_HEADER_RE = re.compile(r"^(?P<indent>[ \t]*)command:[ \t]*$")


def import_compose(text: str) -> dict:
    """Parse one compose file's single service into a settings-shaped dict.

    Returns ``{"args", "env", "container", "notes", "identity", "service",
    "container_name"}``. Raises ``ValueError`` for anything this fleet's
    compose files never produce: invalid YAML, a non-mapping document, a
    missing/empty `services:` block, or more than one service (this
    importer is one-service-per-file by construction — adopting a
    multi-service file would need a decision about which service is "the"
    model, which is out of scope here).
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid compose YAML: {exc}") from exc

    if not isinstance(doc, dict):
        raise ValueError(
            f"compose document must be a mapping, found {type(doc).__name__}"
        )

    services = doc.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError(
            f"compose document must have a non-empty 'services' mapping, found {services!r}"
        )
    if len(services) != 1:
        raise ValueError(
            f"expected exactly one service, found {sorted(services)}"
        )

    service_name, service = next(iter(services.items()))
    if not isinstance(service, dict):
        raise ValueError(
            f"service {service_name!r} must be a mapping, found {type(service).__name__}"
        )

    return {
        "args": _import_args(service.get("command")),
        "env": _import_env(service.get("environment")),
        "container": _import_container(service),
        "notes": _import_notes(text),
        "identity": _import_identity(service.get("volumes")),
        "service": service_name,
        "container_name": service.get("container_name"),
    }


def _import_args(command) -> dict:
    """C1's parser owns every value shape (positional, bare flag,
    multi-value, JSON blob) — this function's only job is turning a compose
    array back into the one shell-quoted line parse_argline expects.

    Compose also allows `command:` as a bare shell string, but every profile
    on this fleet uses the list form; a string here would silently iterate
    character by character rather than token by token, so it is rejected
    loudly instead of guessed at."""
    if not command:
        return {}
    if not isinstance(command, list):
        raise ValueError(
            f"'command' must be a list, found {type(command).__name__}"
        )
    argline = " ".join(shlex.quote(str(token)) for token in command)
    return parse_argline(argline)


def _import_env(environment) -> dict:
    """Compose allows BOTH encodings of `environment:` — a mapping and a
    `KEY=value` list — and nothing constrains this fleet's files to one.

    A shape that is neither raises ValueError, never AttributeError: the
    adopt sweep isolates per-profile failures on ``(ValueError,
    EngineError)`` (app.routers.settings.adopt), and any other exception
    class escapes that isolation into a 500 with earlier profiles' writes
    already committed.

    A list entry with no ``=`` is compose's host-passthrough form. There is
    no host environment to resolve it from at import time, so it imports as
    the empty value rather than being dropped — dropping it would lose the
    operator's only record that the variable is meant to be set at all.
    """
    if not environment:
        return {}
    if isinstance(environment, dict):
        return {str(key): str(value) for key, value in environment.items()}
    if isinstance(environment, list):
        env = {}
        for entry in environment:
            key, sep, value = str(entry).partition("=")
            env[key] = value if sep else ""
        return env
    raise ValueError(
        f"'environment' must be a mapping or a list, "
        f"found {type(environment).__name__}"
    )


def _import_container(service: dict) -> dict:
    """C1's allowlist only (app.settings_store.CONTAINER_ALLOWLIST) —
    `volumes` is a real compose key but is deliberately not Deck-editable,
    so it is excluded here even though it is read separately for identity."""
    return {key: service[key] for key in CONTAINER_ALLOWLIST if key in service}


def _import_identity(volumes) -> str | None:
    """The checkpoint directory name is recoverable ONLY from a host mount
    targeting /model — nothing else in a compose file names it. String-form
    entries only (`host:container[:mode]`); this fleet's compose files never
    use the long (mapping) volume form."""
    if not volumes:
        return None
    for entry in volumes:
        if not isinstance(entry, str):
            continue
        parts = entry.split(":")
        if len(parts) < 2:
            continue
        host, container_path = parts[0], parts[1]
        if container_path == _MODEL_MOUNT_TARGET:
            return Path(host).name
    return None


def _import_notes(text: str) -> dict:
    """Recover `#` lines living inside the `command:` block by re-scanning
    the RAW text, not the parsed document — yaml.safe_load has already
    thrown comments away by the time _import_args runs. A line ends the
    block once it is non-blank, not a comment, and indented no deeper than
    `command:` itself — the same rule that ends any YAML block-list."""
    lines = text.splitlines()
    comments: list[str] = []
    in_block = False
    block_indent = 0

    for line in lines:
        if not in_block:
            match = _COMMAND_HEADER_RE.match(line)
            if match:
                in_block = True
                block_indent = len(match.group("indent"))
            continue

        stripped = line.strip()
        if not stripped:
            continue
        current_indent = len(line) - len(line.lstrip(" \t"))
        if stripped.startswith("#"):
            if current_indent > block_indent:
                comments.append(stripped.lstrip("#").strip())
            continue
        if current_indent <= block_indent:
            in_block = False

    notes: dict = {}
    if comments:
        notes["args"] = "\n".join(comments)
    return notes
