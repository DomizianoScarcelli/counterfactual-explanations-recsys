import os
import warnings
from typing import Generator, List, Optional, Tuple, Union

from recbole.config import Config
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from config import (ALLOWED_MUTATIONS, DATASET, GENERATIONS, HALLOFFAME_RATIO,
                    MODEL, POP_SIZE)
from genetic.dataset.utils import (get_dataloaders,
                                   get_sequence_from_interaction, load_dataset,
                                   save_dataset, train_test_split)
from genetic.genetic import GeneticGenerationStrategy
from genetic.mutations import ReplaceMutation, SwapMutation, parse_mutations
from genetic.utils import NumItems
from models.config_utils import generate_model, get_config
from models.model_funcs import model_predict
from type_hints import GoodBadDataset
from utils import set_seed

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate( interaction: Union[Interaction, Tensor], model:
             SequentialRecommender, alphabet: Optional[List[int]] = None) -> GoodBadDataset:
    """
    Generates the dataset of good and bad points from a sequence in the
    Interaction, using the model as a black box oracle. The dataset can be used
    to train an automata that accepts good sequences and rejects bad sequences
    using the functions in `automata_learning.py`


    Args:
        interaction: the Interaction object containing the sequence for which a
        counterfactual is needed.
        model: the black box model that is used as an oracle to get the
        probability distribution or label of the sequence.
        alphabet: the alphabet that is used in order to perform the mutations.
        If None, then a default alphabet is used
    Returns:
        The dataset in the form of a tuple (good_points, bad_points), where
        good_points and bad_points are lists of LabeledTensors.
    """
    if isinstance(interaction, Interaction):
        sequence = get_sequence_from_interaction(interaction)
    elif isinstance(interaction, Tensor):
        sequence = interaction
    else:
        raise NotImplemented(f"Sequence of type {type(sequence)} is not supported. Tensor or Interaction supported")
    assert (
        sequence.size(0) == 1
    ), f"Only batch size of 1 is supported, sequence shape is: {sequence.shape}"
    # user_id = interaction.interaction["user_id"][0].item()
    
    sequence = sequence.squeeze(0)
    assert len(sequence.shape) == 1, f"Sequence dim must be 1: {
        sequence.shape}"
    allowed_mutations = parse_mutations(ALLOWED_MUTATIONS)
    if alphabet is None:
        alphabet = list(range(NumItems.ML_1M.value))
    good_genetic_strategy = GeneticGenerationStrategy(
        input_seq=sequence,
        predictor=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=POP_SIZE,
        good_examples=True,
        generations=GENERATIONS,
        halloffame_ratio=HALLOFFAME_RATIO,
        alphabet=alphabet
    )
    good_examples = good_genetic_strategy.generate()
    good_examples = good_genetic_strategy.postprocess(good_examples)
    bad_genetic_strategy = GeneticGenerationStrategy(
        input_seq=sequence,
        predictor=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=POP_SIZE,
        good_examples=False,
        generations=GENERATIONS,
        halloffame_ratio=HALLOFFAME_RATIO,
        alphabet=alphabet
    )
    bad_examples = bad_genetic_strategy.generate()
    bad_examples = bad_genetic_strategy.postprocess(bad_examples)
    
    return good_examples, bad_examples


def interaction_generator(config: Config) -> Generator[Interaction, None, None]:
    # raw_interactions = interaction.interaction
    # user_ids = raw_interactions["user_id"]
    # item_ids = raw_interactions["item_id"]
    # item_matrix = raw_interactions["item_id_list"]
    # item_length = raw_interactions["item_length"]
    _, _, test_data = get_dataloaders(config)
    assert test_data is not None, "Test data is None"
    for data in test_data:
        interaction = data[0]
        yield interaction

def sequence_generator(config: Config) -> Generator[Tensor, None, None]:
    _, _, test_data = get_dataloaders(config)
    assert test_data is not None, "Test data is None"
    for data in test_data:
        interaction = data[0]
        yield get_sequence_from_interaction(interaction)


def dataset_generator(
    config: Config, use_cache: bool = True
) -> Generator[
    GoodBadDataset,
    None,
    None,
]:
    interactions = interaction_generator(config)
    model = generate_model(config)
    for i, interaction in enumerate(interactions):
        cache_path = os.path.join(
            f"dataset_cache/interaction_{i}_dataset.pickle"
        )
        if os.path.exists(cache_path) and use_cache:
            dataset = load_dataset(cache_path)
        else:
            dataset = generate(interaction, model)
            if use_cache:
                save_dataset(dataset, cache_path)
        
        yield dataset


if __name__ == "__main__":
    set_seed()
    config = get_config(model=MODEL, dataset=DATASET)
    datasets = dataset_generator(config, use_cache=False)
    for dataset in datasets:
        dataset_save_path = "saved/counterfactual_dataset.pickle"
        save_dataset(dataset, save_path=dataset_save_path)
        print(dataset)
        break
