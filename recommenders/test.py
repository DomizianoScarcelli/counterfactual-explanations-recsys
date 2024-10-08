# from recbole.quick_start import run_recbole
# from recbole.trainer import Trainer
# from recbole.model.sequential_recommender import BERT4Rec
from recbole.utils import init_seed, get_model
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
import torch
import numpy as np
from copy import deepcopy
import pickle
from typing import Union

from dataset_generator import GeneticGenerationStrategy

# Set the config file or pass in model/dataset parameters
parameter_dict_ml1m = {
        'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
        'train_neg_sample_args': None,
        "eval_batch_size": 1
        }
config = Config(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict_ml1m)
# Initialize logger and seed
# init_logger(config)
init_seed(config['seed'], config['reproducibility'])

# Load dataset and pre-trained model
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# Load a pre-trained model checkpoint
model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
checkpoint_file = "saved/Bert4Rec_ml1m.pth"
checkpoint = torch.load(checkpoint_file, map_location=config['device'])
model.load_state_dict(checkpoint["state_dict"])
model.load_other_parameter(checkpoint.get("other_parameter"))

MAX_LENGTH = 50

def predict(model: SequentialRecommender, interaction: Interaction, argmax: bool=True) -> torch.Tensor:
    preds = model.full_sort_predict(interaction)
    if argmax:
        preds = preds.argmax(dim=1)
    # assert preds.size(0) == eval_batch_size
    return preds


# def predict(model: SequentialRecommender, sequence: torch.Tensor, user_id: int) -> int:
#     pass

def model_predict(seq:torch.Tensor, prob: bool=True):
    #Pad with 0s
    seq = torch.cat((seq, torch.zeros((MAX_LENGTH - seq.size(0))))).to(seq.dtype).unsqueeze(0)
    new_interaction = deepcopy(interaction)
    new_interaction.interaction["item_id_list"] = seq
    preds = predict(model, new_interaction, argmax=not prob)
    if not prob:
        return preds.item()
    return preds

def generate_counterfactual_dataset(interaction: Interaction):
    genetic_strategy = GeneticGenerationStrategy()
    sequence = interaction.interaction["item_id_list"] 
    assert sequence.size(0) == 1, f"Only batch size of 1 is supported, sequence shape is: {sequence.shape}"
    user_id = interaction.interaction["user_id"][0].item()

    #Trim zeros
    #TODO: don't know if this is working
    sequence = sequence[:torch.nonzero(sequence, as_tuple=False).max().item() + 1] if sequence.nonzero().size(0) > 0 else torch.tensor([])
    sequence = sequence.squeeze(0)
    assert len(sequence.shape) == 1, f"Sequence dim must be 1: {sequence.shape}"
    good_examples, bad_examples = genetic_strategy.generate(sequence, model=model_predict, clean=True)
    # print([ex[1].item() for ex in good_examples], [ex[1].item() for ex in bad_examples])
    # print(len(good_examples), len(bad_examples))
    return good_examples, bad_examples

def save_dataset(dataset, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

def load_dataset(load_path: str):
    with open(load_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    for i, data in enumerate(test_data):
        # Interaction is an object and the first of the tuple. 
        # if i < 1:
        #     continue
        interaction = data[0]
        # raw_interactions = interaction.interaction
        # user_ids = raw_interactions["user_id"]
        # item_ids = raw_interactions["item_id"]
        # item_matrix = raw_interactions["item_id_list"]
        # item_length = raw_interactions["item_length"]

        # pred = predict_from_interaction(model, interaction)
        good_examples, bad_examples = generate_counterfactual_dataset(interaction)
        dataset_save_path = "saved/counterfactual_dataset.pickle"
        save_dataset((good_examples, bad_examples), save_path=dataset_save_path)
        break




