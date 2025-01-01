import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from config import ConfigParams
from constants import PADDING_CHAR
from generation.utils import random_points_with_offset


class Mutation(ABC):
    def __init__(self):
        self.name: str

    def __call__(self, seq: List[int], alphabet: List[int]) -> Tuple[List[int]]:
        # TODO: junk, remove index

        # Change the seed according to the index of the mutated sequence
        assert PADDING_CHAR not in seq, f"Padding char {PADDING_CHAR} is in seq: {seq}"
        result = (self._apply(seq, alphabet),)
        return result

    @abstractmethod
    def _apply(self, seq: List[int], alphabet: List[int]) -> List[int]:
        pass


class ReplaceMutation(Mutation):
    def __init__(self, num_replaces: int = 1):
        super().__init__()
        self.num_replaces = num_replaces
        self.name = "replace"

    def _apply(self, seq: List[int], alphabet: List[int]) -> List[int]:
        for _ in range(self.num_replaces):
            i = random.sample(range(1, len(seq)), 1)[0]
            new_value = random.choice(alphabet)
            # avoid repetitions
            while new_value in seq:
                new_value = random.choice(alphabet)
            seq[i] = new_value
        return seq


class SwapMutation(Mutation):
    def __init__(self, offset_ratio: float = 0.5):
        super().__init__()
        self.offset_ratio = offset_ratio
        self.name = "swap"

    def _apply(self, seq: List[int], alphabet: List[int]) -> List[int]:
        max_offset = round(len(seq) * self.offset_ratio)
        i, j = random_points_with_offset(len(seq) - 1, max_offset)
        seq[i], seq[j] = seq[j], seq[i]
        return seq


class ReverseMutation(Mutation):
    def __init__(self, offset_ratio: float = 0.5):
        super().__init__()
        self.offset_ratio = offset_ratio
        self.name = "reverse"

    def _apply(self, seq: List[int], alphabet: List[int]) -> List[int]:
        max_offset = round(len(seq) * self.offset_ratio)
        i, j = random_points_with_offset(len(seq) - 1, max_offset)
        seq[i : j + 1] = seq[i : j + 1][::-1]
        return seq


class ShuffleMutation(Mutation):
    def __init__(self, offset_ratio: float = 0.5):
        super().__init__()
        self.offset_ratio = offset_ratio
        self.name = "shuffle"

    def _apply(self, seq: List[int], alphabet: List[int]) -> List[int]:
        max_offset = round(len(seq) * self.offset_ratio)
        i, j = random_points_with_offset(len(seq) - 1, max_offset)
        subseq = seq[i : j + 1]
        random.shuffle(subseq)
        seq[i : j + 1] = subseq
        return seq


class AddMutation(Mutation):
    def __init__(self, num_additions: int = 1):
        super().__init__()
        self.name = "add"
        self.num_additions = num_additions

    def _apply(self, seq: list[int], alphabet: list[int]) -> list[int]:
        for _ in range(self.num_additions):
            random_item = random.choice(alphabet)
            # avoid repetition
            while random_item in seq:
                random_item = random.choice(alphabet)
            i = random.sample(range(1, len(seq)), 1)[0]
            seq.insert(i, random_item)
        return seq


class DeleteMutation(Mutation):
    def __init__(self, num_deletions: int = 1):
        super().__init__()
        self.name = "delete"
        self.num_deletions = num_deletions

    def _apply(self, seq: list[int], alphabet: list[int]) -> list[int]:
        for _ in range(self.num_deletions):
            i = random.sample(range(len(seq)), 1)[0]
            seq.remove(seq[i])
        return seq


def contains_mutation(mutation_type: type, mutations_list: List[Mutation]) -> bool:
    return any(isinstance(m, mutation_type) for m in mutations_list)


def remove_mutation(
    mutation_type: type, mutations_list: List[Mutation]
) -> List[Mutation]:
    return [m for m in mutations_list if not isinstance(m, mutation_type)]


def parse_mutations(muts: List[str]):
    """
    Used to transform a list of mutations names
    (for example take from a config file) to a
    list of mutation classes

    Args:
        muts: List of mutations names.
    """
    return [mut for mut in ALL_MUTATIONS if mut.name in muts]


ALL_MUTATIONS: List[Mutation] = [
    SwapMutation(),
    ReplaceMutation(num_replaces=ConfigParams.NUM_REPLACES),
    ShuffleMutation(),
    ReverseMutation(),
    AddMutation(num_additions=ConfigParams.NUM_ADDITIONS),
    DeleteMutation(num_deletions=ConfigParams.NUM_DELETIONS),
    ReplaceMutation(),
]
