import pytest
from recbole.config import Config
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from config import ConfigParams
from generation.dataset.utils import interaction_to_tensor
from models.config_utils import generate_model, get_config
from models.model_funcs import model_predict
from utils_classes.generators import InteractionGenerator


@pytest.fixture()
def model(config) -> SequentialRecommender:
    return generate_model(config)

@pytest.fixture()
def interaction(config) -> Interaction:
    interaction = next(InteractionGenerator(config)) 
    # print(interaction.interaction)
    return interaction

@pytest.fixture()
def sequence(interaction) -> Tensor:
    return interaction_to_tensor(interaction).squeeze(0)

def test_model_predict(model, sequence):
    print(f"""Executing predict on:
          sequence: {sequence}
          """)
    preds = model_predict(sequence, model, prob=True)
    assert isinstance(preds, Tensor), "Preds are not torch.Tensor"
    print(f"Preds are: {preds} with shape {preds.shape}")

