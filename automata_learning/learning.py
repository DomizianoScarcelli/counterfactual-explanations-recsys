from utils import printd
import os
from pathlib import Path
from typing import List, Tuple, Union

from aalpy.automata.Dfa import Dfa
from aalpy.learning_algs import run_RPNI
from torch import Tensor

from alignment.alignment import augment_constraint_automata
from automata_learning.utils import load_automata
from config import ConfigParams
from generation.dataset.utils import load_dataset
from type_hints import GoodBadDataset


def _generate_automata(
    dataset: List[Tuple[list, bool]],
    load_if_exists: bool = True,
    save_path: str = "automata.pickle",
) -> Union[None, Dfa]:
    """
    Util function that runs the RPNI algorithm over the input dataset, which is
    loaded from cache if specified, and if the file exists.

    Args:
        dataset: the dataset of good and bad points the dataset is learned on.
        load_if_exists: tell the function to load the dataset from cache if it exists.
        save_path: the dataset cache path.

    Returns:
        The learned DFA which accepts good points and rejects bad points.
    """
    if Path(f"saved/saved_automatas/{save_path}").exists() and load_if_exists:
        printd("[INFO] Loaded existing automata", level=1)
        dfa = load_automata(save_path)
        return dfa
    printd(
        "[INFO] Existing automata not found, generating a new one based on the provided dataset"
    )
    dfa = run_RPNI(
        data=dataset,
        automaton_type="dfa",
        print_info=ConfigParams.DEBUG > 0,
        input_completeness="sink_state",
    )
    return dfa  # type: ignore


def generate_automata_from_dataset(
    dataset: GoodBadDataset,
    load_if_exists: bool = True,
    save_path: str = "automata.pickle",
) -> Dfa:
    """
    Given a dataset with the following syntax:
        ([(torch.tensor([...]), good_label), ...],
          [(torch.tensor([...]), bad_label), ...])
    it learns a DFA that accepts good points and rejects bad points
    """
    good_points, bad_points = dataset
    data = [(seq[0].tolist(), True) for seq in good_points] + [
        (seq[0].tolist(), False) for seq in bad_points
    ]
    dfa = _generate_automata(data, load_if_exists, save_path)
    if dfa is None:
        raise RuntimeError("DFA is None, aborting")
    assert dfa.is_input_complete(), "Dfa is not input complete"
    return dfa


def generate_single_accepting_sequence_dfa(sequence):
    """
    Generates a DFA that only accepts the input sequence.
    """
    raise NotImplementedError("Deprecated, remove")


def learning_pipeline(source: List[Tensor] | List[int], dataset: GoodBadDataset) -> Dfa:
    if isinstance(source[0], Tensor):
        source = [c.item() for c in source]  # type: ignore

    assert isinstance(source[0], int)
    a_dfa = generate_automata_from_dataset(dataset, load_if_exists=False)
    a_dfa_aug = augment_constraint_automata(a_dfa, source)  # type: ignore

    # NOTE: debug info about what the automata is doing
    good_label = dataset[0][0][1]
    bad_label = dataset[1][0][1]
    printd(
        f"[DEBUG INFO] Once inverted, the automata will accept traces with the label similar to {bad_label} (target), and reject labels similar to {good_label} (starting point)"
    )
    return a_dfa_aug


if __name__ == "__main__":
    printd("Generating automata from saved dataset")

    # Remove non-determinism
    dataset = load_dataset(load_path=Path("saved/counterfactual_dataset.pickle"))
    # dataset = make_deterministic(dataset)

    dfa = generate_automata_from_dataset(dataset, load_if_exists=False)
    # dfa.visualize()
