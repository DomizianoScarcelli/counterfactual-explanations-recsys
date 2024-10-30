import pickle
from typing import Set, Tuple

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import init_seed
from torch import Tensor
from torch.utils.data import DataLoader

from models.utils import pad
from type_hints import Dataset, GoodBadDataset


def train_test_split(dataset: Dataset, test_split: float = 0.2):
    train, test = [], []
    train_end = round(len(dataset) * (1-test_split))
    for i in range(train_end):
        train.append(dataset[i])
    for i in range(train_end, len(dataset)-1):
        test.append(dataset[i])
    return train, test


def save_dataset(dataset: Tuple[GoodBadDataset, GoodBadDataset], save_path: str):
    """
    Saves the dataset as a .pickle file

    Args:
        dataset: The dataset that has to be saved.
        save_path: The save path
    """
    with open(save_path, "wb") as f:
        print(f"Dataset saved to {save_path}")
        pickle.dump(dataset, f)


def load_dataset(load_path: str) -> Tuple[GoodBadDataset, GoodBadDataset]:
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


def get_sequence_from_interaction(interaction: Interaction) -> Tensor:
    sequence = interaction.interaction["item_id_list"]
    length = interaction.interaction["item_length"]
    # print(f"""Interaction-Sequence info: 
    #       Sequence: {sequence} 
    #       Length: {length}
    #       Unpadded: {sequence[:, :length]}
    #       """)

    # Changes padding character from 0 to -1
    unpadded = sequence[:, :length].flatten()
    return pad(unpadded, sequence.size(-1)).unsqueeze(0)

def get_dataset_alphabet(dataset: GoodBadDataset) -> Set[int]:
    alphabet = set()
    good, bad = dataset
    for example in good + bad:
        alphabet |= set(example[0].tolist())
    return alphabet

