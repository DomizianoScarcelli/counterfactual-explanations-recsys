from copy import deepcopy
from typing import List

import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor


def predict(model: SequentialRecommender, seq: Tensor, argmax: bool=True) -> torch.Tensor:
    """Returns the prediction of the model on the interaction.

    Args:
        model: the SequentialRecommender which is in charge of the prediction
        interaction: the Interaction on which the prediction has to be made on
        argmax: if true, it performs argmax on the result to return the label,
        otherwise it returns the raw logits

    Returns:
        raw logits if argmax is False, label otherwise
    """
    preds = model.full_sort_predict(seq)
    if argmax:
        preds = preds.argmax(dim=1)
    return preds

def batch_predict(model: SequentialRecommender, seq: Tensor, argmax: bool=True) -> torch.Tensor:
    """Returns the prediction of the model on the interaction.

    Args:
        model: the SequentialRecommender which is in charge of the prediction
        interaction: the Interaction on which the prediction has to be made on
        argmax: if true, it performs argmax on the result to return the label,
        otherwise it returns the raw logits

    Returns:
        raw logits if argmax is False, label otherwise
    """
    preds = model.full_sort_predict_from_sequence(seq)
    if argmax:
        preds = preds.argmax(dim=1)
    return preds

def model_predict(seq:torch.Tensor, 
                  model: SequentialRecommender,
                  prob: bool=True):
    preds = batch_predict(model=model, seq=seq, argmax=not prob)
    if not prob:
        return preds.item()
    return preds



def model_batch_predict(batch_seq:torch.Tensor, 
                  interactions: List[Interaction], 
                  model: SequentialRecommender,
                  prob: bool=True):
    batch_size = batch_seq.size(0)
    assert len(interactions) == batch_size, f"Interaction len must be equal to batch size {len(interactions)} != {batch_size}"
    preds = torch.tensor([])
    for batch in range(batch_size):
        seq = batch_seq[batch,:]
        pred = predict(model=model, seq=seq, argmax=not prob)
        preds = torch.stack((preds, pred), dim=0)
    return preds

