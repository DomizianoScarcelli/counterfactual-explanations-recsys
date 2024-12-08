from constants import cat2id
import json
import os
import pickle
import random
from enum import Enum
from statistics import mean
from typing import List, TypedDict

import _pickle as cPickle
from recbole.data.dataset.sequential_dataset import SequentialDataset
from torch import Tensor

from config import ConfigParams
from constants import PADDING_CHAR
from type_hints import Dataset, RecDataset
from utils_classes.Cached import Cached
from utils_classes.distances import edit_distance


class ItemInfo(TypedDict):
    name: str
    category: List[str]


class NumItems(Enum):
    ML_100K = 1682
    ML_1M = 3703
    MOCK = 6


class Items(Enum):
    MOCK = set(range(1, 7))
    ML_1M = os.path.join("data", "universe.txt")


class Category(Enum):
    ML_1M = os.path.join("data", "category_map.json")


def get_items(dataset: RecDataset=ConfigParams.DATASET):
    # TODO: You remove this an take the alphabet from the category map, or from the id2token keys
    # in this way you can remove the universe.txt files, which are not so much elegant.
    def load_items(path):
        with open(path, "r") as f:
            return f.read()

    if dataset == RecDataset.ML_1M:
        items = Items.ML_1M

    if isinstance(items, Items):
        value = items.value
    else:
        value = items

    if isinstance(value, set):
        return value
    elif isinstance(value, str):
        # Use Data class to handle file caching
        data = Cached(value, load_fn=load_items).get_data()
        item_set = set(
            int(x) for x in data.replace("{", "").replace("}", "").split(",")
        ) - {PADDING_CHAR}
        # category_map_keys = set(get_category_map(ConfigParams.DATASET).keys())

        # print(f"Item set length: {len(item_set)}\n Category map length: {len(category_map_keys)}\n Intersection length: {len(item_set & category_map_keys)}")
        # return item_set & category_map_keys
        return item_set
    else:
        raise ValueError("items must be a set or a path to a set")


def get_category_map(dataset: RecDataset=ConfigParams.DATASET):

    def load_json(path):
        with open(path, "r") as f:
            data = json.load(f)

        return {int(key): value for key, value in data.items()}

    if dataset == RecDataset.ML_1M:
        category = Category.ML_1M
    else:
        raise NotImplementedError(
            f"get_category_map not implemented for dataset {dataset}"
        )

    return Cached(category.value, load_fn=load_json).get_data()

def label2cat(label: int, dataset: RecDataset=ConfigParams.DATASET, encode: bool=False):
    category_map = get_category_map(dataset)
    categories = category_map[label]
    if not encode:
        return categories
    return [cat2id[cat] for cat in categories]

def get_remapped_dataset(dataset: RecDataset) -> SequentialDataset:
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    if dataset == RecDataset.ML_1M:
        path = os.path.join("data", "ml-1m-SequentialDataset.pth")
    else:
        raise NotImplementedError(
            f"get_category_map not implemented for dataset {dataset}"
        )

    return Cached(path, load_fn=load_pickle).get_data()


def id2token(dataset: RecDataset, id: int) -> int:
    """
    Maps interal item ids to external tokens
    """
    remapped_dataset = get_remapped_dataset(dataset)
    return int(remapped_dataset.id2token("item_id", ids=id))


def token2id(dataset: RecDataset, token: str) -> int:
    """
    Maps external item tokens to internal ids.
    """
    remapped_dataset: SequentialDataset = get_remapped_dataset(dataset)
    return int(remapped_dataset.token2id("item_id", tokens=token))


def get_item_info(datset: RecDataset, id: int) -> ItemInfo:
    """
    Returns the information about a certain item in the dataset
    """
    # TODO:implement
    info = {"name": "", "category": []}
    return info


def clone(x):
    return cPickle.loads(cPickle.dumps(x))


def random_points_with_offset(max_value: int, max_offset: int):
    i = random.randint(1, max_value - 1)
    j = random.randint(max(0, i - max_offset), min(max_value - 1, i + max_offset))
    # Sort i and j to ensure i <= j
    return tuple(sorted([i, j]))


def _evaluate_generation(input_seq: Tensor, dataset: Dataset, label: int):
    # Evaluate label
    same_label = sum(1 for ex in dataset if ex[1] == label)
    # Evaluate example similarity
    distances_norm = []
    distances_nnorm = []
    for seq, _ in dataset:
        distances_norm.append(edit_distance(input_seq, seq))
        distances_nnorm.append(edit_distance(input_seq, seq, normalized=False))
    return (same_label / len(dataset)), (mean(distances_norm), mean(distances_nnorm))
