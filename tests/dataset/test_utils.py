import pytest
import torch

from genetic.dataset.utils import dataset_difference
from type_hints import Dataset


def test_dataset_difference():
    mock_dataset: Dataset = [(torch.tensor([1,3,2]),True),
         (torch.tensor([1,2,3]),True),
         (torch.tensor([2,3,1,5]),True),
         (torch.tensor([5,4,1]),True)]

    mock_other: Dataset = [(torch.tensor([1,3,2]),True),
         (torch.tensor([2,3,1,5]),True),
         (torch.tensor([5,4,1]),True)]

    expected = [([1,2,3],True)]

    difference = dataset_difference(mock_dataset, mock_other)

    difference_tolist = [(t.squeeze(0).tolist(), l) for t, l in difference]
    
    assert expected == difference_tolist, f"Result is wrong"
