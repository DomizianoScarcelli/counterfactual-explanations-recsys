import pickle
from typing import Set, Tuple

from numpy._core.multiarray import MAXDIMS
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import init_seed
from torch import Tensor
from torch.utils.data import DataLoader

from constants import MAX_LENGTH, PADDING_CHAR
from models.utils import pad, pad_batch, replace_padding
from type_hints import Dataset, GoodBadDataset


def save_dataset(dataset: GoodBadDataset, save_path: str):
    """
    Saves the dataset as a .pickle file

    Args:
        dataset: The dataset that has to be saved.
        save_path: The save path
    """
    with open(save_path, "wb") as f:
        print(f"Dataset saved to {save_path}")
        pickle.dump(dataset, f)


def load_dataset(load_path: str) -> GoodBadDataset:
    """
    Loads the dataset from disk.

    Args:
        load_path: The path from which to load the dataset
    Returns:
        The dataset (good_points, bad_points).
    """
    with open(load_path, "rb") as f:
        print(f"Dataset loaded from {load_path}")
        return pickle.load(f)


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates the train, val, test dataloaders from the dataset in the config file.

    Args:
        config: the Config file, which contains the dataset used to create the pytorch dataloaders.
    Returns:
        The train, val, test torch Dataloaders.
    """
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return train_data, valid_data, test_data


def make_deterministic(dataset: Tuple[Dataset, Dataset]) -> Tuple[Dataset, Dataset]:
    g, b = dataset
    print(f"Dataset len before making it deterministic: {len(g)}, {len(b)}")
    new_g, new_b = [], []
    ids = set()
    for p, l in g:
        if tuple(p.tolist()) in ids:
            continue
        new_g.append((p, l))
        ids.add(tuple(p.tolist()))
    for p, l in b:
        if tuple(p.tolist()) in ids:
            continue
        new_b.append((p, l))
        ids.add(tuple(p.tolist()))
    dataset = (new_g, new_b)
    print(f"Dataset len after making it deterministic: {
          len(new_g)}, {len(new_b)}")
    return dataset


def interaction_to_tensor(interaction: Interaction) -> Tensor:
    """
    Given an interaction object, it returns the (batched) sequences padded with
    the ConfigParams.PADDIN_CHAR.
    """
    sequence = interaction.interaction["item_id_list"]
    # length = interaction.interaction["item_length"]
    if sequence.dim() == 1 and sequence.size(0) == MAX_LENGTH:
        sequence = sequence.unsqueeze(0)
    batch_size = sequence.size(0)
    
    assert sequence.shape == (batch_size, MAX_LENGTH), f"Seq has incorrect shape: {sequence.shape} != {(batch_size, MAX_LENGTH)}"

    return replace_padding(sequence, 0, PADDING_CHAR)

def get_dataset_alphabet(dataset: GoodBadDataset) -> Set[int]:
    alphabet = set()
    good, bad = dataset
    for example in good + bad:
        alphabet |= set(example[0].tolist())
    return alphabet

def dataset_difference(dataset: Dataset, other: Dataset) -> Dataset:
    """
    Returns all the entries from one dataset that do not appear in the other
    dataset, effectively doing an equivalent of the set difference operation.

    Args:
        dataset: The original dataset
        other: The datset that will be substracted to the original dataset

    Returns:
        A new dataset such that (dataset INTERSECTION other) is the empty set.
    """
    dataset_map = {tuple(row.squeeze(0).tolist()): label for row, label in dataset}
    other_map = {tuple(row.squeeze(0).tolist()): label for row, label in other}

    dataset_keys = set(dataset_map.keys())
    other_keys = set(other_map.keys())

    difference_keys = dataset_keys - other_keys

    difference_dataset = [(torch.tensor(key).unsqueeze(0), dataset_map[key]) for key in difference_keys]
    return difference_dataset

def are_dataset_equal(dataset: Dataset, other: Dataset) -> bool:
    dataset_map = {tuple(row.squeeze(0).tolist()): label for (row, label) in dataset}
    other_map = {tuple(row.squeeze(0).tolist()): label for (row, label) in other}

    dataset_keys = set(dataset_map.keys())
    other_keys = set(other_map.keys())
    return dataset_keys == other_keys


