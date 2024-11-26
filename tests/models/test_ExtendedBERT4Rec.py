import pytest
import torch
from torch import Tensor
from tqdm import tqdm
from genetic.dataset.utils import interaction_to_tensor

from genetic.dataset.utils import interaction_to_tensor
from models.extended_models.ExtendedBERT4Rec import ExtendedBERT4Rec
from models.model_funcs import model_predict
from utils_classes.generators import InteractionGenerator

@pytest.fixture()
def batched_sequences(interactions) -> Tensor:
    sequences = torch.tensor([])
    for i in interactions:
        seq = interaction_to_tensor(i)
        sequences = torch.cat((sequences, seq), dim=0)
    return sequences

class TestPredictFromSequence:
    def test_BatchPred_IsEqualToStackedSinglePreds(self, model: ExtendedBERT4Rec, batched_sequences: Tensor):
        """
        Tests if the predictions on the batch sequences gives the same result
        as getting one prediction at the time.
        """
        batch_preds = model(batched_sequences)
        for i, seq in enumerate(tqdm(batched_sequences, "Batched full sort predict test...")):
            pred = model(seq.unsqueeze(0)).squeeze()
            label = pred.argmax(-1).item()
            batch_label = batch_preds[i].argmax(-1).item()
            assert label == batch_label, f"Labels are different! {label} != {batch_label}"
            assert torch.allclose(pred, batch_preds[i])

    def test_BatchPred_AreAllEqual_WhenSequenceIsTheSame(self, model: ExtendedBERT4Rec, batched_sequences: Tensor):
        """
        Test if the batch prediction produces a tensor with the same label when
        the input tensor to the model consists of the same tensor repeated N
        times.
        """
        for seq in tqdm(batched_sequences, leave=False):
            batch_size = 128
            batch = seq.unsqueeze(0).repeat(batch_size, 1)
            preds = model(batch)
            label_set = {x.argmax(-1).item() for x in preds}
            assert len(label_set) == 1, f"Label for the same sequence are not equal! Uniques labels are: {label_set}"

    def test_PredictFromInteraction_IsEqualToPredictFromSequence(self, model: ExtendedBERT4Rec, interactions: InteractionGenerator):
        for interaction in interactions:
            sequence = interaction_to_tensor(interaction)
            model_int = model(interaction)
            model_seq = model(sequence)
            label_int = model_int.argmax(-1).item()
            label_seq = model_seq.argmax(-1).item()
            assert label_int == label_seq, f"Label are different {label_int} != {label_seq}"
            assert torch.allclose(model_int, model_seq), f"Logits are different"

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
