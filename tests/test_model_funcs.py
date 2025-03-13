import pytest
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from core.generation.dataset.utils import interaction_to_tensor
from core.models.config_utils import generate_model
from core.models.model_funcs import model_predict
from utils.generators import InteractionGenerator


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

