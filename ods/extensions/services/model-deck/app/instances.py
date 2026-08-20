"""Pure helpers for deck-created engine instances (INST I1). No I/O, no
engine names — kinds come in as strings and go to app.engine_kinds."""


def check_observed_gpus(gpu_indices: list[int], observed: list[int] | None) -> None:
    """E1 debt 3 (design §10): a declared GPU must be one the node OBSERVES.
    Unknown pool → refuse (never coerce a declaration against a guess)."""
    if observed is None:
        raise ValueError("node GPUs are unobserved right now; cannot validate gpu_indices")
    missing = sorted(set(gpu_indices) - set(observed))
    if missing:
        raise ValueError(f"gpu_indices {missing} not observed on this node "
                         f"(observed: {sorted(observed)})")
