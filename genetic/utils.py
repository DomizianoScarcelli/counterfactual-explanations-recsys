import random
from copy import deepcopy
from enum import Enum

import _pickle as cPickle
import Levenshtein
import torch.nn.functional as F
from torch import Tensor

from type_hints import Dataset
from utils import set_seed


class NumItems(Enum):
    ML_100K=1682
    ML_1M=3703
    MOCK=6


def clone(x):
    # return deepcopy(x)
    return cPickle.loads(cPickle.dumps(x))

def edit_distance(t1: Tensor, t2: Tensor):
    str1, str2 = str(t1), str(t2) #Levenshtein.ratio only works with strings
    # return 1 - Levenshtein.ratio(str1, str2)
    return Levenshtein.distance(t1.tolist(), t2.tolist())

def cosine_distance(prob1: Tensor, prob2: Tensor) -> float:
    return 1 - F.cosine_similarity(prob1, prob2, dim=-1).item()

def self_indicator(seq1, seq2):
    if len(seq1) != len(seq2):
        return 0
    return float("inf") if (seq1 == seq2).all() else 0

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
