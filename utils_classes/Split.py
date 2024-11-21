from __future__ import annotations
from typing import List, Tuple
from type_hints import SplitTuple, TraceSplit

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
    def __init__(self, *args: int | float):
        self.split = tuple(args)
        assert len(self.split) == 3
        types = set(type(s) for s in self.split)
        assert len(types) == 1, "Split should have all ints (in case of absolute values) or floats (in case of ratios)"
        if list(types)[0] == int:
            self.is_ratio = False
        elif list(types)[0] == float:
            self.is_ratio = True
        else:
            raise ValueError(f"Split must contains ints or floats, not {list(types)[0]}")

        if self.is_ratio:
            assert sum(self.split) == 1, f"Ratio should sum to 1, not {sum(self.split)}"

    def is_coherent(self, seq: List[int]) -> bool:
        """ It tells us if the split is coherent with the input sequence, meaning if the sequence can actually be splitted according to the split.

        Args:
            seq: The sequence that has to be splitted

        Returns:
            True if the sequence can be splitted with the defined split, False otherwhise.
        """
        split = self.split
        if self.is_ratio:
            split = self.to_abs(seq).split
        if len(seq) != sum(split):
            return False
        
        # TODO: Are other checks necessary?
        return True

    def to_ratio(self, seq: List[int]) -> Split:
        assert not self.is_ratio, "Split is already a ratio"
        return Split(*tuple(i/len(seq) for i in self.split))
    
    def to_abs(self, seq: List[int]) -> Split:
        assert self.is_ratio, "Split is already absolute"
        return Split(*tuple(round(i * len(seq)) for i in self.split))

    def apply(self, seq: List[int]) -> TraceSplit:
        assert self.is_coherent(seq)
        split = self.split
        if self.is_ratio:
            split = self.to_abs(seq).split
        start, middle, _ = split
        executed = seq[:start]
        mutable = seq[start:start+middle]
        fixed = seq[start+middle:]
        return executed, mutable, fixed
    
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
