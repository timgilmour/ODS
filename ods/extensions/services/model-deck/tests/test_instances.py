"""Tests for app.instances — pure helpers for deck-created engine instances
(INST I1). This file currently pins only `check_observed_gpus` (Task 2 /
E1 debt 3); Task 7 adds the rest of the module."""

import pytest

from app.instances import check_observed_gpus


def test_check_observed_gpus_accepts_a_subset_and_names_the_miss():
    check_observed_gpus([2, 4], [2, 3, 4])
    with pytest.raises(ValueError, match=r"gpu_indices \[5\] not observed on this node \(observed: \[2, 3, 4\]\)"):
        check_observed_gpus([2, 5], [2, 3, 4])


def test_check_observed_gpus_refuses_when_the_pool_is_unknown():
    with pytest.raises(ValueError, match="unobserved right now"):
        check_observed_gpus([2], None)
