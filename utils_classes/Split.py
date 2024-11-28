from __future__ import annotations

from typing import List, Tuple

from exceptions import SplitNotCoherent
from type_hints import TraceSplit


class Split:
    """
    Utility class that defines methods useful to modify, convert and manage a
    `SplitTuple`, which is a a tuple structured as (Executed, Mutable,
                                                             Fixed) where each
    part of the tuple defines the number of elements belonging to it. 

    If a sequence of length 50 has a split (10, 20, 20), it means that the
    first 10 elements are executed, the next 20 are mutable andd the last 20
    are fixed.
    """
    def __init__(self, *args: int | float | None):
        self.split = tuple(args)
        self.parsed = None not in self.split
        assert len(self.split) == 3
        types = set(type(s) for s in self.split if s is not None) # it can contain None, which will be parsed in the `parse_nan` method
        assert len(types) == 1, "Split should have all ints (in case of absolute values) or floats (in case of ratios)"
        if list(types)[0] == int:
            self.is_ratio = False
        elif list(types)[0] == float:
            self.is_ratio = True
        else:
            raise ValueError(f"Split must contains ints or floats, not {list(types)[0]}")

        if self.is_ratio:
            if None not in self.split:
                assert sum(self.split) == 1, f"Ratio should sum to 1, not {sum(self.split)}" #type: ignore
            else:
                assert sum(s for s in self.split if s is not None) <= 1

    def _parse_single_nan(self, seq: List[int]) -> Split:
        values = [s for s in self.split if s is not None]
        if self.is_ratio:
            assert 0 <= sum(values) <= 1
        else:
            assert 0 <= sum(values) <= len(seq)

        if self.is_ratio:
            inferredNone = (1 - sum(values))
            return Split(*tuple(inferredNone if s is None else s for s in self.split))
        inferredNone = (len(seq) - sum(values))
        return Split(*tuple(inferredNone if s is None else s for s in self.split))

    def _parse_double_nan(self, seq: List[int]) -> Split:
        # If the value is int, it should be between 0 and len(seq)
        # If the value is float, it shold be between 0 and 1
        value = [s for s in self.split if s is not None][0]
        if self.is_ratio:
            assert 0 <= value <= 1
        else:
            assert 0 <= value <= len(seq)

        # A None value means that the value has to be inferred. Since we have a value
        # meaning that inferredNone + inferredNone + value = len(seq) (or 1 if ratio)
        if self.is_ratio:
            inferredNone = (1 - value) / 2
            return Split(*tuple(inferredNone if s is None else s for s in self.split))
        inferredNone = (len(seq) - value) // 2
        split = [inferredNone if s is None else s for s in self.split]
        if sum(split) < len(seq):
            nan_index = [i for i, s in enumerate(self.split) if s is None][0]
            split[nan_index] += 1
        return Split(*split)

    def parse_nan(self, seq:List[int]) -> Split:
        """ 
        Given a split with 1 or 2 None values, it infers those None values
        based on the input sequence.

        In particular if there is a single None value, it's filled with the
        value that sums to 1 (if ratio) or to the sequence length (if not ratio)

        If there are two Nones, it's filled with the value that sould reach 1
        (if ratio) or the sequence length (if not ratio) divided by two and
        rounded up to the nearest integer (a plus one may be added to one of
        the two values because of rounding)

        Args:
            seq: The sequence on which the split has to be inferred on.

        Returns:
            A split coherent with the sequence where there aren't any None
            values.
        """
        # If None is present, it should appear in exactly two positions 
        nans = [s for s in self.split if s is None]
        if len(nans) == 0:
            print("Split doesn't contain any None values, returning it")
            return self
        if len(nans) == 1:
            self.parsed = True
            return self._parse_single_nan(seq)
        if len(nans) == 2:
            self.parsed = True
            return self._parse_double_nan(seq)
        else:
            raise ValueError(f"Cannot infer split with {len(nans)} None values")

        
    def is_coherent(self, seq: List[int]) -> bool:
        """ 
        It tells us if the split is coherent with the input sequence, meaning if the sequence can actually be splitted according to the split.

        Args:
            seq: The sequence that has to be splitted

        Returns:
            True if the sequence can be splitted with the defined split, False otherwhise.
        """
        split = self.split

        assert isinstance(seq, List)
        if self.is_ratio:
            split = self.to_abs(seq).split


        if len(seq) != sum(split): #type: ignore
            return False
        
        # TODO: Are other checks necessary?
        return True

    def to_ratio(self, seq: List[int]) -> Split:
        assert not self.is_ratio, "Split is already a ratio"
        return Split(*tuple(i/len(seq) for i in self.split)) #type: ignore
    
    def to_abs(self, seq: List[int]) -> Split:
        assert self.is_ratio, "Split is already absolute"
        assert self.parsed, f"Split contains None value, call .parse_nan(seq) first"
        return Split(*tuple(round(i * len(seq)) for i in self.split)) #type: ignore

    def apply(self, seq: List[int]) -> TraceSplit:
        self = self.parse_nan(seq)
        if not self.is_coherent(seq):
            raise SplitNotCoherent(f"Sequence length is not coherent with split: {len(seq)} < {len(self)}")
        split = self.split
        if self.is_ratio:
            split = self.to_abs(seq).split
        start, middle, _ = split
        executed = seq[:start]
        mutable = seq[start:start+middle] #type: ignore
        fixed = seq[start+middle:] #type: ignore
        return executed, mutable, fixed
    
    def __len__(self) -> int:
        return sum(self.split) #type: ignore

    def __eq__(self, other) -> bool:
        if not isinstance(other, Split):
            return False
        if isinstance(other, Tuple):
            other = Split(*other)
        return self.split == other.split

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.split)

    def __hash__(self):
        return hash(self.split)
