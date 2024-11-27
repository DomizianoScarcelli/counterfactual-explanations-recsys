import random
from enum import Enum
import os

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

def get_items(items: Items):
    if isinstance(items.value, set):
        return items.value
    elif isinstance(items.value, str):
        with open(items.value, "r") as f:
            return set(int(x) for x in f.read().replace("{", "").replace("}", "").split(",")) - {PADDING_CHAR}
    else:
        raise ValueError("items must be a set of ar a path to a set")

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
    distances = []
    for seq, _ in dataset:
        distances.append(edit_distance(input_seq, seq))
    return (same_label / len(dataset)), (sum(distances)/len(distances))

