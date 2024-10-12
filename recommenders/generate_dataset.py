from recbole.utils import init_seed, get_model
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
import torch
import numpy as np
from copy import deepcopy
import pickle
from typing import Union, Optional, Generator
from enum import Enum
from deap_generator import GeneticGenerationStrategy
from recommenders.model_funcs import model_predict


class RecDataset(Enum):
    ML_1M = "ml-1m"

class RecModel(Enum):
    BERT4Rec = "BERT4Rec"

def generate_counterfactual_dataset(interaction: Interaction, model: SequentialRecommender):
    sequence = interaction.interaction["item_id_list"] 
    assert sequence.size(0) == 1, f"Only batch size of 1 is supported, sequence shape is: {sequence.shape}"
    user_id = interaction.interaction["user_id"][0].item()

    #Trim zeros
    #TODO: don't know if this is working
    sequence = sequence[:torch.nonzero(sequence, as_tuple=False).max().item() + 1] if sequence.nonzero().size(0) > 0 else torch.tensor([])
    sequence = sequence.squeeze(0)
    assert len(sequence.shape) == 1, f"Sequence dim must be 1: {sequence.shape}"
    good_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                      predictor=lambda x:
                                                      model_predict(seq=x,
                                                                    interaction=interaction,
                                                                    model=model,
                                                                    prob=True),
                                                      pop_size=2000,
                                                      good_examples=True,
                                                      generations=10)
    good_examples = good_genetic_strategy.generate()
    good_examples = good_genetic_strategy.postprocess(good_examples)
    bad_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                     predictor=lambda x:
                                                     model_predict(seq=x,
                                                                   interaction=interaction,
                                                                   model=model,
                                                                   prob=True),
                                                     pop_size=2000,
                                                     good_examples=False,
                                                     generations=10)
    bad_examples = bad_genetic_strategy.generate()
    bad_examples = bad_genetic_strategy.postprocess(bad_examples)
    return good_examples, bad_examples

def save_dataset(dataset, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

def load_dataset(load_path: str):
    with open(load_path, "rb") as f:
        return pickle.load(f)


def load_data(config):
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return train_data, valid_data, test_data 

def generate_model(config: Config):
    train_data, _, _= load_data(config)
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    checkpoint_file = "saved/Bert4Rec_ml1m.pth"
    checkpoint = torch.load(checkpoint_file, map_location=config['device'])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    return model

def get_config(dataset: RecDataset, model: RecModel):
    parameter_dict_ml1m = {
            'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
            'train_neg_sample_args': None,
            "eval_batch_size": 1}
    return Config(model=model.value, dataset=dataset.value, config_dict=parameter_dict_ml1m)

def interaction_generator(model: SequentialRecommender, config: Config) -> Generator[Interaction, None, None]:
    # raw_interactions = interaction.interaction
    # user_ids = raw_interactions["user_id"]
    # item_ids = raw_interactions["item_id"]
    # item_matrix = raw_interactions["item_id_list"]
    # item_length = raw_interactions["item_length"]
    _, _, test_data = load_data(config)
    assert test_data is not None, "Test data is None"
    for data in test_data:
        interaction = data[0]
        yield interaction

if __name__ == "__main__":
    config = get_config(model=RecModel.BERT4Rec, dataset=RecDataset.ML_1M)
    model = generate_model(config)
    interactions = interaction_generator(model, config)
    for interaction in interactions:
        good_examples, bad_examples = generate_counterfactual_dataset(interaction, model)
        dataset_save_path = "saved/counterfactual_dataset.pickle"
        save_dataset((good_examples, bad_examples), save_path=dataset_save_path)
        break #generate just one dataset




