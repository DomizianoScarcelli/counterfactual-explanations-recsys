from recbole.quick_start import run_recbole
from recbole.utils import init_seed, init_logger, get_model
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.model.sequential_recommender import BERT4Rec
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
import torch
import numpy as np
from typing import List
from copy import deepcopy

from dataset_generator import GeneticGenerationStrategy

# Set the config file or pass in model/dataset parameters
eval_batch_size = 1
config = Config(model='BERT4Rec', dataset='ml-100k', config_dict={"train_neg_sample_args": None, "eval_batch_size": eval_batch_size})
# Initialize logger and seed
# init_logger(config)
init_seed(config['seed'], config['reproducibility'])

# Load dataset and pre-trained model
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# Load a pre-trained model checkpoint
model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

def predict(model: SequentialRecommender, interaction: Interaction) -> torch.Tensor:
    preds = model.full_sort_predict(interaction).argmax(dim=1)
    assert preds.size(0) == eval_batch_size
    return preds


# def predict(model: SequentialRecommender, sequence: torch.Tensor, user_id: int) -> int:
#     pass

def generate_counterfactual_dataset(interaction: Interaction) -> List[Interaction]:
    genetic_strategy = GeneticGenerationStrategy()
    sequence = interaction.interaction["item_id_list"] 
    assert sequence.size(0) == 1, f"Only batch size of 1 is supported, sequence shape is: {sequence.shape}"
    user_id = interaction.interaction["user_id"][0].item()
    def model_predict(seq):
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        seq = seq.unsqueeze(0)
        new_interaction = deepcopy(interaction)
        new_interaction.interaction["item_id_list"] = seq
        return predict(model, new_interaction)
    counterfactual_dataset = genetic_strategy.generate(sequence[0].numpy(), model=model_predict)
    print(counterfactual_dataset )

for data in test_data:
    # Interaction is an object and the first of the tuple. 
    interaction = data[0]
    # raw_interactions = interaction.interaction
    # user_ids = raw_interactions["user_id"]
    # item_ids = raw_interactions["item_id"]
    # item_matrix = raw_interactions["item_id_list"]
    # item_length = raw_interactions["item_length"]

    # pred = predict_from_interaction(model, interaction)
    generate_counterfactual_dataset(interaction)
    break




