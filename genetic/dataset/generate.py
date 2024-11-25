import warnings
from typing import List, Optional, Union

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from config import ConfigParams
from genetic.dataset.utils import get_sequence_from_interaction
from genetic.genetic import GeneticGenerationStrategy
from genetic.mutations import parse_mutations
from genetic.utils import NumItems
from models.model_funcs import model_predict
from type_hints import GoodBadDataset

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate(interaction: Union[Interaction, Tensor], 
             model: SequentialRecommender, 
             alphabet: Optional[List[int]] = None) -> GoodBadDataset:
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
    allowed_mutations = parse_mutations(ConfigParams.ALLOWED_MUTATIONS)
    if alphabet is None:
        alphabet = list(range(NumItems.ML_1M.value))
    good_genetic_strategy = GeneticGenerationStrategy(
        input_seq=sequence,
        predictor=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=ConfigParams.POP_SIZE,
        good_examples=True,
        generations=ConfigParams.GENERATIONS,
        halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
        alphabet=alphabet
    )
    good_examples = good_genetic_strategy.generate()
    good_examples = good_genetic_strategy.postprocess(good_examples)
    bad_genetic_strategy = GeneticGenerationStrategy(
        input_seq=sequence,
        predictor=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=ConfigParams.POP_SIZE,
        good_examples=False,
        generations=ConfigParams.GENERATIONS,
        halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
        alphabet=alphabet
    )
    bad_examples = bad_genetic_strategy.generate()
    bad_examples = bad_genetic_strategy.postprocess(bad_examples)
    
    return good_examples, bad_examples

# if __name__ == "__main__":
#     set_seed()
#     config = get_config(model=MODEL, dataset=DATASET)
#     datasets = DatasetGenerator(config, use_cache=False)
#     for dataset in datasets:
#         dataset_save_path = "saved/counterfactual_dataset.pickle"
#         save_dataset(dataset, save_path=dataset_save_path)
#         print(dataset)
#         break
