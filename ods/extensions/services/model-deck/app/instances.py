"""Pure helpers for deck-created engine instances (INST I1). No I/O, no
engine names — kinds come in as strings and go to app.engine_kinds."""

from app.engine_kinds import gpu_indices_of
from app.engines import GuardError


def next_resource_name(kind: str, taken: set[str]) -> str:
    """The lowest-numbered `{kind}-{n}` (n >= 1) not already in `taken`.
    "slot0" never enters this scheme at all — it is the serving-slot's own
    reserved OBSERVATION key (a swap node's UI card id, `app/observe.py`'s
    `slot_key` -> `f"{node_id}/slot0"`), never a `{kind}-{n}` name a create
    call could produce; declaring an engine literally called "slot0" is
    refused upstream at validation (`app/engine_kinds.py`'s
    `validate_engines`, ~:324-348), not here."""
    n = 1
    while f"{kind}-{n}" in taken:
        n += 1
    return f"{kind}-{n}"


def allocate_port(port_range: dict, taken: set[int]) -> int:
    """The lowest free host port in `port_range` (D-I1-3: allocated from the
    node entry's `instance_port_range`, no scanning beyond it). GuardError,
    never a silent wraparound or reuse, when the range is exhausted."""
    for p in range(port_range["start"], port_range["end"] + 1):
        if p not in taken:
            return p
    raise GuardError(f"no free port in instance_port_range {port_range['start']}-{port_range['end']}")


def instance_document(entry: dict) -> dict:
    """The wire document `InstancesClient.request` sends for one instance
    entry — reads the GPU claim through `gpu_indices_of`, the ONE accessor
    (D-I1-2), never the legacy scalar field directly."""
    return {"resource": entry["resource"], "kind": entry["kind"],
            "gpu_indices": gpu_indices_of(entry), "port": entry["port"], "env": dict(entry["env"])}


def check_observed_gpus(gpu_indices: list[int], observed: list[int] | None) -> None:
    """E1 debt 3 (design §10): a declared GPU must be one the node OBSERVES.
    Unknown pool → refuse (never coerce a declaration against a guess)."""
    if observed is None:
        raise ValueError("node GPUs are unobserved right now; cannot validate gpu_indices")
    missing = sorted(set(gpu_indices) - set(observed))
    if missing:
        raise ValueError(f"gpu_indices {missing} not observed on this node "
                         f"(observed: {sorted(observed)})")
