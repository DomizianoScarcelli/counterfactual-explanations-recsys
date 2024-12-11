import torch

from config import ConfigParams
from generation.dataset.utils import (are_dataset_equal, dataset_difference,
                                   get_dataloaders, interaction_to_tensor)
from models.utils import replace_padding
from type_hints import Dataset
from utils_classes.generators import InteractionGenerator

# class TestDataloaderTypes:
#     def test_GetDataloaders_ReturnsCorrectType_WhenLeaveOneOut(self, config):
#         print(f"\n------LEAVE ONE OUT------\n")
#         train, val, test = get_dataloaders(config, split_strategy="LS")
#         for train_ex in train:
#             print(f"LS Train example:", train_ex.interaction)
#             break
#         for val_ex in val:
#             print(f"LS Val example:", val_ex[0].interaction)
#             break
#         for test_ex in test:
#             print(f"LS Test example:", test_ex[0].interaction)
#             break
    
#     def test_GetDataloaders_ReturnsCorrectType_WhenRandomSplit(self, config):
#         print(f"\n------RANDOM SPLIT------\n")
#         train, val, test = get_dataloaders(config, split_strategy="RS")
#         for train_ex in train:
#             print(f"RS Train example:", train_ex.interaction)
#             break
#         for val_ex in val:
#             print(f"RS Val example:", val_ex[0].interaction)
#             break
#         for test_ex in test:
#             print(f"RS Test example:", test_ex[0].interaction)
#             break

class TestGetSequenceFromInteraction:
    def test_GetSequenceFromInteraction_IsCorrect(self, interactions: InteractionGenerator):
        for i, interaction in enumerate(interactions):
            if i < len(interactions.data)-1:
                seqs = interaction_to_tensor(interaction)
                assert seqs.size(0) == ConfigParams.TEST_BATCH_SIZE

class TestReplacePadding:
    def test_replacePadding_replaceCorrectly_whenSequenceNotBatched(self):
        sequences = torch.tensor([
            [1, 2, 3, -1, -1],  
            [4, 5, 6, 7, -1],  
            [8, 9, 10, 11, 12], 
            [13, -1, -1, -1, -1] 
        ])

        expected_sequences = torch.tensor([
            [1, 2, 3, 0, 0],  
            [4, 5, 6, 7, 0],  
            [8, 9, 10, 11, 12], 
            [13, 0, 0, 0, 0 ] 
        ])

        for i, padded_seq in enumerate(sequences):
            assert torch.all(replace_padding(padded_seq, -1, 0) == expected_sequences[i])

    def test_replacePadding_replaceCorrectly_whenSequenceBatched(self):
        sequences = torch.tensor([
            [1, 2, 3, -1, -1],  
            [4, 5, 6, 7, -1],  
            [8, 9, 10, 11, 12], 
            [13, -1, -1, -1, -1] 
        ])

        expected_sequences = torch.tensor([
            [1, 2, 3, 0, 0],  
            [4, 5, 6, 7, 0],  
            [8, 9, 10, 11, 12], 
            [13, 0, 0, 0, 0 ] 
        ])
        result = replace_padding(sequences, -1, 0)
        assert torch.all(result == expected_sequences)

def test_DatasetDifference_IsCorrect():
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

def test_AreDatasetEqual_isCorrect():
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


