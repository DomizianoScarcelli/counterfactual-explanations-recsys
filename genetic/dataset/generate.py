import os
import warnings
from typing import Generator, Tuple, Union

from recbole.config import Config
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from config import DATASET, MODEL
from genetic.dataset.utils import (get_dataloaders,
                                   get_sequence_from_interaction, load_dataset,
                                   save_dataset, train_test_split)
from genetic.genetic import GeneticGenerationStrategy
from genetic.mutations import Mutations
from recommenders.config_utils import generate_model, get_config
from recommenders.model_funcs import model_predict
from type_hints import GoodBadDataset
from utils import set_seed

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate(
    interaction: Union[Interaction, Tensor], model: SequentialRecommender
) -> Tuple[
    Tuple[GoodBadDataset, GoodBadDataset], Tuple[GoodBadDataset, GoodBadDataset]
]:
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
    Returns:
        The dataset in the form of a tuple (good_points, bad_points), where
        good_points and bad_points are lists of LabeledTensors.
    """
    if isinstance(interaction, Interaction):
        sequence = get_sequence_from_interaction(interaction)
    elif isinstance(interaction, Tensor):
        sequence = interaction
    assert (
        sequence.size(0) == 1
    ), f"Only batch size of 1 is supported, sequence shape is: {sequence.shape}"
    # user_id = interaction.interaction["user_id"][0].item()

    # Trim zeros
    sequence = sequence.squeeze(0)
    assert len(sequence.shape) == 1, f"Sequence dim must be 1: {
        sequence.shape}"
    allowed_mutations = [Mutations.SWAP, Mutations.REPLACE]
    good_genetic_strategy = GeneticGenerationStrategy(
        input_seq=sequence,
        predictor=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=2000,
        good_examples=True,
        generations=10,
    )
    good_examples = good_genetic_strategy.generate()
    good_examples = good_genetic_strategy.postprocess(good_examples)
    bad_genetic_strategy = GeneticGenerationStrategy(
        input_seq=sequence,
        predictor=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=2000,
        good_examples=False,
        generations=10,
    )
    bad_examples = bad_genetic_strategy.generate()
    bad_examples = bad_genetic_strategy.postprocess(bad_examples)

    train_good, test_good = train_test_split(good_examples)
    train_bad, test_bad = train_test_split(bad_examples)
    train_dataset = (train_good, train_bad)
    test_dataset = (test_good, test_bad)
    return train_dataset, test_dataset


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


def dataset_generator(
    config: Config, use_cache: bool = True
) -> Generator[
    Tuple[Tuple[GoodBadDataset, GoodBadDataset],
          Tuple[GoodBadDataset, GoodBadDataset]],
    None,
    None,
]:
    interactions = interaction_generator(config)
    model = generate_model(config)
    for i, interaction in enumerate(interactions):
        train_cache_path = os.path.join(
            f"dataset_cache/interaction_{i}_dataset_train.pickle"
        )
        test_cache_path = os.path.join(
            f"dataset_cache/interaction_{i}_dataset_test.pickle"
        )
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(test_cache_path)
            and use_cache
        ):
            train = load_dataset(train_cache_path)
            test = load_dataset(test_cache_path)
        else:
            train, test = generate(interaction, model)
            if use_cache:
                save_dataset(train, train_cache_path)
                save_dataset(test, test_cache_path)

        yield train, test


if __name__ == "__main__":
    set_seed()
    config = get_config(model=MODEL, dataset=DATASET)
    datasets = dataset_generator(config, use_cache=False)
    for dataset in datasets:
        dataset_save_path = "saved/counterfactual_dataset.pickle"
        save_dataset(dataset, save_path=dataset_save_path)
        break
