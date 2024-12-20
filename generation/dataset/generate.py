import warnings
from typing import List, Optional, Union

from recbole.trainer import Interaction
from torch import Tensor

from generation.dataset.utils import interaction_to_tensor
from generation.strategies.abstract_strategy import GenerationStrategy
from type_hints import GoodBadDataset

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate(
    interaction: Union[Interaction, Tensor],
    good_strat: Optional[GenerationStrategy],
    bad_strat: Optional[GenerationStrategy],
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
        if good_strat:
            good_strat.replace_alphabet(alphabet)
        if bad_strat:
            bad_strat.replace_alphabet(alphabet)

    good_examples, bad_examples = [], []
    if good_strat:
        good_examples = good_strat.generate()
    if bad_strat:
        bad_examples = bad_strat.generate()

    assert (
        good_strat is not None or bad_strat is not None
    ), "Unexpected error, good strat OR bad strat should be not None"
    # print(
    #     f"=================================GOOD DATASET================================="
    # )
    # for _, v in good_examples:
    #     print(f"Good {v}")
    # print(
    #     f"=================================BAD DATASET================================="
    # )
    # for _, v in bad_examples:
    #     print(f"Bad {v}")
    return good_examples, bad_examples
