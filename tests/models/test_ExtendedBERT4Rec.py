from typing import List

import pytest
import torch
from recbole.trainer import Interaction
from torch import Tensor

from genetic.dataset.generate import interaction_generator
from genetic.dataset.utils import get_sequence_from_interaction
from models.extended_models.ExtendedBERT4Rec import ExtendedBERT4Rec


@pytest.fixture()
def interactions(config, batch_size: int=16) -> List[Interaction]:
    interactions = []
    generator = interaction_generator(config)
    while len(interactions) < batch_size:
        interactions.append(next(generator))
    return interactions


@pytest.fixture()
def sequences(interactions) -> Tensor:
    sequences = torch.tensor([])
    for i in interactions:
        seq = get_sequence_from_interaction(i)
        sequences = torch.cat((sequences, seq), dim=0)
    batch_size = len(interactions)
    assert sequences.size(0) == batch_size, f"Sequences shape {sequences.shape} not matching batch size {batch_size}"
    return sequences

@pytest.mark.incremental
class TestPredictFromSequence:
    
    def test_full_sort_predict_from_sequence(self, model: ExtendedBERT4Rec, sequences):
        for single_seq in sequences:
            single_seq = single_seq.unsqueeze(0)
            assert single_seq.size(0) == 1, f"single seq must have shape [1, length], {single_seq.shape}"
            pred = model.full_sort_predict(single_seq)
            print(f"[test_full_sort_predict_from_sequence] pred is: {pred} with shape {pred.shape}")
   
    def test_batched_full_sort_predict(self, model: ExtendedBERT4Rec, sequences):
        pred = model.full_sort_predict(sequences)
        print(f"[test_full_sort_predict_from_sequence] pred is: {pred} with shape {pred.shape}")

    def test_determination(self, model: ExtendedBERT4Rec, sequences):
        single_seq = sequences[0].unsqueeze(0)
        assert single_seq.size(0) == 1, f"single seq must have shape [1, length], {single_seq.shape}"
        first_pred = model.full_sort_predict_from_sequence(single_seq)
        second_pred = model.full_sort_predict_from_sequence(single_seq)
        assert torch.allclose(first_pred, second_pred)
