import random
from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from genetic.utils import random_points_with_offset


class Mutation(ABC):
    @abstractmethod
    def __call__(self, seq: List[int], alphabet: List[int]) -> List[int]:
        pass


class ReplaceMutation(Mutation):
    def __init__(self, num_replaces: int = 1):
        self.num_replaces = num_replaces
        self.name = "replace"

    def __call__(self, seq: List[int], alphabet: List[int]) -> List[int]:
        for _ in range(self.num_replaces):
            i = random.sample(range(len(seq)), 1)[0]
            new_value = random.choice(alphabet)
            while new_value in seq:
                new_value = random.choice(alphabet)
            seq[i] = new_value
        return seq

class SwapMutation(Mutation):
    def __init__(self, offset_ratio: float = 0.8):
        self.offset_ratio = offset_ratio
        self.name = "swap"

    def __call__(self, seq: List[int], alphabet: List[int]) -> List[int]:
        max_offset = round(len(seq) * self.offset_ratio)
        i, j = random_points_with_offset(len(seq) - 1, max_offset)
        seq[i], seq[j] = seq[j], seq[i]
        return seq

class ReverseMutation(Mutation):
    def __init__(self, offset_ratio: float = 0.8):
        self.offset_ratio = offset_ratio
        self.name = "reverse"

    def __call__(self, seq: List[int], alphabet: List[int]) -> List[int]:
        max_offset = round(len(seq) * self.offset_ratio)
        i, j = random_points_with_offset(len(seq)-1, max_offset)
        seq[i:j+1] = seq[i:j+1][::-1]
        return seq

class ShuffleMutation(Mutation):
    def __init__(self, offset_ratio: float = 0.8):
        self.offset_ratio = offset_ratio
        self.name = "shuffle"

    def __call__(self, seq: List[int], alphabet: List[int]) -> List[int]:
        max_offset = round(len(seq) * self.offset_ratio)
        i, j = random_points_with_offset(len(seq)-1, max_offset)
        subseq = seq[i:j+1]  
        random.shuffle(subseq) 
        seq[i:j+1] = subseq  
        return seq


class AddMutation(Mutation):
    def __init__(self):
        self.name = "add"

    def __call__(self, seq: list[int], alphabet: list[int]) -> list[int]:
        random_item = random.choice(alphabet)
        while random_item in seq:
            random_item = random.choice(alphabet)
        i = random.sample(range(len(seq)), 1)[0]
        seq.insert(i, random_item)
        return seq

class DeleteMutation(Mutation):
    def __init__(self):
        self.name = "delete"

    def __call__(self, seq: list[int], alphabet: list[int]) -> list[int]:
        i = random.sample(range(len(seq)), 1)[0]
        seq.remove(seq[i])
        return seq


def contains_mutation(mutation_type: type, mutations_list: List[Mutation]) -> bool:
    return any(isinstance(m, mutation_type) for m in mutations_list)

def remove_mutation(mutation_type: type, mutations_list: List[Mutation]) -> List[Mutation]:
    return [m for m in mutations_list if not isinstance(m, mutation_type)]

ALL_MUTATIONS: List[Mutation] = [SwapMutation(), ReplaceMutation(),
                                 ShuffleMutation(), ReverseMutation(),
                                 AddMutation(), DeleteMutation(),
                                 ReplaceMutation()]
