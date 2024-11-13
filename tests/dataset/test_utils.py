import torch

from genetic.dataset.utils import are_dataset_equal, dataset_difference
from type_hints import Dataset


def test_dataset_difference():
    mock_dataset: Dataset = [(torch.tensor([1,3,2]),True),
         (torch.tensor([1,2,3]),True),
         (torch.tensor([2,3,1,5]),True),
         (torch.tensor([5,4,1]),True)]

    mock_dataset_2: Dataset = [(torch.tensor([1,3,2]),True),
         (torch.tensor([1,2,3]),True),
         (torch.tensor([2,3,1,5]),True)]

    mock_dataset_3: Dataset = [(torch.tensor([1,3,2]),True),
         (torch.tensor([2,3,1,5]),True),
         (torch.tensor([5,4,1]),True)]

    expected13 = [([1,2,3],True)]
    expected32 = [([5,4,1],True)]

    difference13 = dataset_difference(mock_dataset, mock_dataset_3)
    difference31 = dataset_difference(mock_dataset_3, mock_dataset)
    difference32 = dataset_difference(mock_dataset_3, mock_dataset_2)

    difference13_tolist = [(t.squeeze(0).tolist(), l) for t, l in difference13]
    difference32_tolist = [(t.squeeze(0).tolist(), l) for t, l in difference32]
    
    assert expected13 == difference13_tolist, f"Result is wrong"
    assert expected32 == difference32_tolist, f"Result is wrong"
    assert [] == difference31, f"Result is wrong"

def test_are_dataset_equal():
    mock_dataset: Dataset = [(torch.tensor([1,3,2]),True),
                             (torch.tensor([1,2,3]),521),
                             (torch.tensor([2,3,1,5]),423),
                             (torch.tensor([5,4,1]),10)]
    mock_dataset_permuted: Dataset = [(torch.tensor([1,3,2]),True),
                                      (torch.tensor([2,3,1,5]),423),
                                      (torch.tensor([5,4,1]),10),
                                      (torch.tensor([1,2,3]),521)]
    mock_other_same_len: Dataset = [(torch.tensor([1,3,2]),True),
                                    (torch.tensor([1,2,5]),132),
                                    (torch.tensor([2,3,1,5]),124),
                                    (torch.tensor([5,4,1]),124)]

    mock_other: Dataset = [(torch.tensor([1,3,2]),124),
                           (torch.tensor([2,3,1,5]),142),
                           (torch.tensor([5,4,1]),152)]

    assert are_dataset_equal(mock_dataset, mock_dataset)

    assert are_dataset_equal(mock_dataset, mock_dataset_permuted)
    assert are_dataset_equal(mock_dataset_permuted, mock_dataset)

    assert not are_dataset_equal(mock_dataset, mock_other)
    assert not are_dataset_equal(mock_other, mock_dataset)
    assert not are_dataset_equal(mock_dataset, mock_other_same_len)
    assert not are_dataset_equal(mock_other_same_len, mock_dataset)


