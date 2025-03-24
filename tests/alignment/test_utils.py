import heapq
from unittest.mock import Mock

import pytest
import torch

from config.constants import MAX_LENGTH
from core.alignment.utils import (  # Replace 'your_module' with the actual module name
    get_path_statistics, postprocess_alignment, prune_paths_by_length,
    syncable)


def test_postprocess_alignment_ReturnsPaddedTensor_WhenAlignedLengthLessThanMax():
    aligned = [1, 2, 3]
    result = postprocess_alignment(aligned)
    assert result.shape == (1, MAX_LENGTH)
    assert isinstance(result, torch.Tensor)


def test_postprocess_alignment_ReturnsTensor_WhenAlignedLengthEqualsMax():
    aligned = [i for i in range(MAX_LENGTH)]
    result = postprocess_alignment(aligned)
    assert result.shape == (1, MAX_LENGTH)
    assert isinstance(result, torch.Tensor)


def test_postprocess_alignment_RaisesError_WhenAlignedLengthGreaterThanMax():
    aligned = [i for i in range(MAX_LENGTH + 1)]
    with pytest.raises(ValueError):
        postprocess_alignment(aligned)

def test_get_path_statistics_ReturnsCorrectStats_WhenPathsGiven():
    paths = [
        (1, 0, None, None, [1, 2, 3], None, None),
        (2, 1, None, None, [4, 5], None, None),
    ]
    stats = get_path_statistics(paths)
    assert stats["num_paths"] == 2
    assert stats["min_cost"] == 1
    assert stats["max_cost"] == 2
    assert stats["num_paths_near_max_length"] >= 0


def test_prune_paths_by_length_PrunesPaths_WhenPathsExceedLimit():
    paths = [(i, i, None, None, None, None, None) for i in range(200_000)]
    pruned_paths = prune_paths_by_length(paths, max_paths=100_000)
    assert len(pruned_paths) == 100_000
    assert heapq.nsmallest(1, pruned_paths)[0][0] == 0


def test_syncable_ReturnsTrue_WhenCharIsSyncable():
    state = Mock()
    state.state_id = 1
    syncable_dict = {1: {42}}
    assert syncable(state, 42, syncable_dict) is True


def test_syncable_ReturnsFalse_WhenCharIsNotSyncable():
    state = Mock()
    state.state_id = 1
    syncable_dict = {1: {99}}
    assert syncable(state, 42, syncable_dict) is False
