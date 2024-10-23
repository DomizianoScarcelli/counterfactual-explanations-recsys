from genetic.utils import NumItems, random_points_with_offset
from enum import Enum
from typing import List
import random

def mutate_replace(seq: List[int], max_value:NumItems=NumItems.ML_1M, num_replaces:int=1):
    for _ in range(num_replaces):
        i = random.sample(range(len(seq)), 1)[0]
        new_value = random.randint(1, max_value.value)
        while (new_value in seq):
            new_value = random.randint(1, max_value.value)
        seq[i] = new_value
    return seq,

def mutate_swap(seq: List[int], offset_ratio: float=0.8):
    max_offset = round(len(seq) * offset_ratio)
    i, j = random_points_with_offset(len(seq)-1, max_offset)
    seq[i], seq[j] = seq[j], seq[i]
    return seq,

def mutate_reverse(seq: List[int], offset_ratio:float=0.3):
    max_offset = round(len(seq) * offset_ratio)
    i, j = random_points_with_offset(len(seq)-1, max_offset)
    seq[i:j+1] = seq[i:j+1][::-1]
    return seq,

# Mutation: Shuffles a random subsequence
def mutate_shuffle(seq: List[int], offset_ratio:float=0.3):
    max_offset = round(len(seq) * offset_ratio)
    i, j = random_points_with_offset(len(seq)-1, max_offset)
    subseq = seq[i:j+1]  
    random.shuffle(subseq) 
    seq[i:j+1] = subseq  
    return seq,

def mutate_add(seq: List[int], max_value: NumItems=NumItems.ML_1M):
    random_item = random.randint(1, max_value.value)
    while random_item in seq:
        random_item = random.randint(1, max_value.value)
    i = random.sample(range(len(seq)), 1)[0]
    seq.insert(i, random_item)
    return seq,

def mutate_delete(seq: List[int]):
    i = random.sample(range(len(seq)), 1)[0]
    seq.remove(seq[i])
    return seq,


class Mutations(Enum):
    SWAP = mutate_swap
    REPLACE = mutate_replace
    SHUFFLE = mutate_shuffle
    REVERSE = mutate_reverse
    ADD = mutate_add
    DELETE = mutate_delete

ALL_MUTATIONS: List[Mutations] = [Mutations.SWAP, Mutations.REPLACE,
                                  Mutations.SHUFFLE, Mutations.REVERSE,
                                  Mutations.ADD, Mutations.DELETE]
