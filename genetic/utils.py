from recbole.data.dataset.sequential_dataset import SequentialDataset
from config import ConfigParams
from utils_classes.Cached import Cached
import os
import random
import json
import pickle
from enum import Enum
from statistics import mean

import _pickle as cPickle
from torch import Tensor

from constants import PADDING_CHAR
from type_hints import Dataset, RecDataset
from utils_classes.distances import edit_distance


class NumItems(Enum):
    ML_100K=1682
    ML_1M=3703
    MOCK=6

class Items(Enum):
    MOCK=set(range(1, 7))
    ML_1M=os.path.join("data", "universe.txt")

class Category(Enum):
    ML_1M=os.path.join("data", "category_map.json")

def get_items(items: Items|str):

    def load_items(path):
        with open(path, "r") as f:
            return f.read()

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

def get_category_map(dataset: RecDataset):

    def load_json(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {int(key): value for key, value in data.items()}

    if dataset == RecDataset.ML_1M:
        category = Category.ML_1M
    else:
        raise NotImplementedError(f"get_category_map not implemented for dataset {dataset}")
        
    return Cached(category.value, load_fn=load_json).get_data()

def get_remapped_dataset(dataset: RecDataset):
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    if dataset == RecDataset.ML_1M:
        path = os.path.join("data", "ml-1m-SequentialDataset.pth")
    else:
        raise NotImplementedError(f"get_category_map not implemented for dataset {dataset}")
        
    return Cached(path, load_fn=load_pickle).get_data()

def id2token(dataset: RecDataset, id: int) -> int:
    remapped_dataset: SequentialDataset = get_remapped_dataset(dataset) 
    return int(remapped_dataset.id2token("item_id", ids=id))

def token2id(dataset: RecDataset, token: str) -> int:
    remapped_dataset: SequentialDataset = get_remapped_dataset(dataset) 
    return int(remapped_dataset.token2id("item_id", tokens=token))

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

