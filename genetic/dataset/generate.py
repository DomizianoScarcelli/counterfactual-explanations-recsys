import warnings
from typing import List, Optional, Union

from recbole.trainer import Interaction
from torch import Tensor

from genetic.abstract_generation import GenerationStrategy
from genetic.dataset.utils import interaction_to_tensor
from type_hints import GoodBadDataset

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate(
    interaction: Union[Interaction, Tensor],
    good_strat: GenerationStrategy,
    bad_strat: GenerationStrategy,
    alphabet: Optional[List[int]] = None,
) -> GoodBadDataset:
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
        sequence = interaction_to_tensor(interaction)
    elif isinstance(interaction, Tensor):
        sequence = interaction
    else:
        raise NotImplemented(
            f"Sequence of type {type(sequence)} is not supported. Tensor or Interaction supported"
        )
    assert (
        sequence.size(0) == 1
    ), f"Only batch size of 1 is supported, sequence shape is: {sequence.shape}"
    if alphabet:
        good_strat.replace_alphabet(alphabet)
        bad_strat.replace_alphabet(alphabet)
    good_examples = good_strat.generate()
    bad_examples = bad_strat.generate()
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
