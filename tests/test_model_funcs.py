from copy import deepcopy

import pytest
import torch
from recbole.config import Config
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from genetic.dataset.generate import interaction_generator
from genetic.dataset.utils import get_sequence_from_interaction
from recommenders.model_funcs import model_batch_predict, model_predict
from recommenders.config_utils import generate_model


@pytest.fixture()
def config():
    parameter_dict_ml1m = {
        'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
        'train_neg_sample_args': None,
        "eval_batch_size": 1
    }
    return Config(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict_ml1m)

@pytest.fixture()
def model(config) -> SequentialRecommender:
    return generate_model(config)

@pytest.fixture()
def interaction(config) -> Interaction:
    interaction = next(interaction_generator(config)) 
    # print(interaction.interaction)
    return interaction

@pytest.fixture()
def sequence(interaction) -> Tensor:
    return get_sequence_from_interaction(interaction).squeeze(0)

def test_model_predict(model, interaction, sequence):
    print(f"""Executing predict on:
          sequence: {sequence}
          """)
    preds = model_predict(sequence, interaction, model, prob=True)
    assert isinstance(preds, Tensor), "Preds are not torch.Tensor"
    print(f"Preds are: {preds} with shape {preds.shape}")

