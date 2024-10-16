from copy import deepcopy
from typing import List

import pytest
import torch
from numpy import single
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Interaction
from torch import Tensor

from models.ExtendedBERT4Rec import ExtendedBERT4Rec
from recommenders.generate_dataset import (generate_model,
                                           get_sequence_from_interaction,
                                           interaction_generator)
from recommenders.model_funcs import model_batch_predict, model_predict


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

def test_full_sort_predict_from_sequence(model: ExtendedBERT4Rec, sequences):
    single_seq = sequences[0].unsqueeze(0)
    assert single_seq.size(0) == 1, f"single seq must have shape [1, length], {single_seq.shape}"
    pred = model._full_sort_predict_from_sequence(single_seq)
    print(f"[test_full_sort_predict_from_sequence] pred is: {pred} with shape {pred.shape}")

def test_batched_full_sort_predict(model: ExtendedBERT4Rec, sequences):
    pred = model.batched_full_sort_predict(sequences)
    print(f"[test_full_sort_predict_from_sequence] pred is: {pred} with shape {pred.shape}")

def test_determination(model: ExtendedBERT4Rec, sequences):
    single_seq = sequences[0].unsqueeze(0)
    assert single_seq.size(0) == 1, f"single seq must have shape [1, length], {single_seq.shape}"
    first_pred = model._full_sort_predict_from_sequence(single_seq)
    second_pred = model._full_sort_predict_from_sequence(single_seq)
    assert torch.allclose(first_pred, second_pred)
