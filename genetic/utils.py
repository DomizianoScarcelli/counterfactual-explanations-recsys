from utils_classes.Cached import Cached
import os
import random
import json
from enum import Enum
from statistics import mean

import _pickle as cPickle
from torch import Tensor

from constants import PADDING_CHAR
from type_hints import Dataset
from utils_classes.distances import edit_distance


class NumItems(Enum):
    ML_100K=1682
    ML_1M=3703
    MOCK=6

class Items(Enum):
    MOCK=set(range(1, 7))
    ML_1M=os.path.join("data", "universe.txt")

class Category(Enum):
    ML_1M=os.path.join("data", "category_mapping.json")

def get_items(items: Items):

    def load_items(path):
        with open(path, "r") as f:
            return f.read()

    if isinstance(items.value, set):
        return items.value
    elif isinstance(items.value, str):
        # Use Data class to handle file caching
        data = Cached(items.value, load_fn=load_items).get_data()
        return set(
            int(x) for x in data.replace("{", "").replace("}", "").split(",")
        ) - {PADDING_CHAR}
    else:
        raise ValueError("items must be a set or a path to a set")

def get_category_map(category: Category):

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    return Cached(category.value, load_fn=load_json).get_data()

def clone(x):
    # return deepcopy(x)
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

