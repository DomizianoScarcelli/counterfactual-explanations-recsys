import random
from enum import Enum

import _pickle as cPickle
import Levenshtein
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from models.utils import pad


class NumItems(Enum):
    ML_100K=1682
    ML_1M=3703
    MOCK=6


def cPickle_clone(x):
    # return deepcopy(x)
    return cPickle.loads(cPickle.dumps(x))

def edit_distance(str1, str2):
    return 1 - Levenshtein.ratio(str1, str2)

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
