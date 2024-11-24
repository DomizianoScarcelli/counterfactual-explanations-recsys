from typing import List

import pytest
import torch
from recbole.trainer import Interaction
from torch import Tensor
from tqdm import tqdm

from genetic.dataset.utils import get_sequence_from_interaction
from models.extended_models.ExtendedBERT4Rec import ExtendedBERT4Rec
from models.model_funcs import model_predict
from utils_classes.generators import InteractionGenerator


@pytest.fixture()
def interactions(config, batch_size: int=16) -> List[Interaction]:
    interactions = []
    generator = InteractionGenerator(config)
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

class TestPredictFromSequence:
    def test_batched_full_sort_predict(self, model: ExtendedBERT4Rec, sequences: Tensor):
        """
        Tests if the predictions on the batch sequences gives the same result
        as getting one prediction at the time.
        """
        batch_preds = model(sequences)

        for i, seq in enumerate(tqdm(sequences, "Batched full sort predict test...")):
            pred = model(seq.unsqueeze(0)).squeeze()
            label = pred.argmax(-1).item()
            batch_label = batch_preds[i].argmax(-1).item()
            assert label == batch_label, f"Labels are different! {label} != {batch_label}"
            # assert torch.all(pred == batch_preds[i])

class TestModelDeterminism:
    def test_model_determinism(self, model: ExtendedBERT4Rec, sequences):
        """
        Tests model determinism, meaning the same model should produce the same
        output when inputted with the same source sequence
        """
        for i, seq in tqdm(enumerate(sequences)):
            if i == 50:
                break
            assert seq.size(0) == 1, f"single seq must have shape [1, length], {seq.shape}"
            first_pred = model_predict(seq, model, prob=True)
            second_pred = model_predict(seq, model, prob=True)

            assert isinstance(first_pred, Tensor)
            assert isinstance(second_pred, Tensor)

            first_label = first_pred.argmax(-1).item()
            second_label = second_pred.argmax(-1).item()
            assert first_label == second_label
            assert torch.all(first_pred == second_pred)


    def test_model_batch_determinism(self, model: ExtendedBERT4Rec, sequences):
        """
        Tests model determinism when the sequences are batched, meaning the same
        model should produce the same output when inputted with the same source
        sequence
        """
        seqs = torch.empty((50, 50))
        for i, seq in tqdm(enumerate(sequences)):
            if i >= 50:
                break
            seqs[i] = seq.squeeze(0)
        seqs = seqs.to(torch.int64)
        pred1 = model_predict(seqs, model, prob=True)
        pred2 = model_predict(seqs, model, prob=True)
        assert torch.all(pred1 == pred2)
