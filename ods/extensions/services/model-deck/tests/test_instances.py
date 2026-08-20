"""Tests for app.instances — pure helpers for deck-created engine instances
(INST I1). Task 2 pinned `check_observed_gpus`; Task 7 adds the allocator
helpers (resource naming, port allocation, the instance document)."""

import pytest

from app.engines import GuardError
from app.instances import (
    allocate_port,
    check_observed_gpus,
    instance_document,
    next_resource_name,
)


def test_check_observed_gpus_accepts_a_subset_and_names_the_miss():
    check_observed_gpus([2, 4], [2, 3, 4])
    with pytest.raises(ValueError, match=r"gpu_indices \[5\] not observed on this node \(observed: \[2, 3, 4\]\)"):
        check_observed_gpus([2, 5], [2, 3, 4])


def test_check_observed_gpus_refuses_when_the_pool_is_unknown():
    with pytest.raises(ValueError, match="unobserved right now"):
        check_observed_gpus([2], None)


def test_next_resource_name_is_lowest_free():
    # No "never slot0" case belongs here: "slot0" is the serving slot's own
    # reserved OBSERVATION key, not a `{kind}-{n}` name this function could
    # ever produce for any real kind (comfyui/hipfire/lemonade/sglang-omni)
    # — see next_resource_name's docstring and app/engine_kinds.py's
    # validate_engines, which is what actually refuses a declared "slot0".
    assert next_resource_name("hipfire", set()) == "hipfire-1"
    assert next_resource_name("hipfire", {"hipfire-1", "hipfire-3"}) == "hipfire-2"


def test_allocate_port_lowest_free_or_guard_error():
    assert allocate_port({"start": 11500, "end": 11502}, {11500}) == 11501
    with pytest.raises(GuardError, match="no free port in instance_port_range 11500-11501"):
        allocate_port({"start": 11500, "end": 11501}, {11500, 11501})


def test_instance_document_reads_the_claim_through_the_accessor():
    e = {"resource": "gguf-a", "kind": "lemonade", "gpu_index": 3, "port": 11500, "env": {"A": "b"},
         "connection": {}, "managed": True}
    assert instance_document(e) == {"resource": "gguf-a", "kind": "lemonade", "gpu_indices": [3], "port": 11500, "env": {"A": "b"}}
