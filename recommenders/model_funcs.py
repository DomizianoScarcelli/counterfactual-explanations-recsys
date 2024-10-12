import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from copy import deepcopy

def predict(model: SequentialRecommender, interaction: Interaction, argmax: bool=True) -> torch.Tensor:
    """Returns the prediction of the model on the interaction.

    Args:
        model: the SequentialRecommender which is in charge of the prediction
        interaction: the Interaction on which the prediction has to be made on
        argmax: if true, it performs argmax on the result to return the label,
        otherwise it returns the raw logits

    Returns:
        raw logits if argmax is False, label otherwise
    """
    preds = model.full_sort_predict(interaction)
    if argmax:
        preds = preds.argmax(dim=1)
    return preds

def model_predict(seq:torch.Tensor, 
                  interaction: Interaction, 
                  model: SequentialRecommender,
                  prob: bool=True):
    #Pad with 0s
    MAX_LENGTH = 50
    seq = torch.cat((seq, torch.zeros((MAX_LENGTH - seq.size(0))))).to(seq.dtype).unsqueeze(0)
    new_interaction = deepcopy(interaction)
    new_interaction.interaction["item_id_list"] = seq
    preds = predict(model, new_interaction, argmax=not prob)
    if not prob:
        return preds.item()
    return preds
