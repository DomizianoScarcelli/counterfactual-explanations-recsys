from utils_classes.Split import Split
import pytest

def test_split_creation():
    seq = [1,2,3,4,5,6,7,8,9,10]
    split = Split(5, 1, 4)
    assert not split.is_ratio
    ex, mut, fix = split.apply(seq)
    assert ex == [1,2,3,4,5]
    assert mut == [6]
    assert fix == [7,8,9,10]

    split = Split(5, 1, 5)
    assert not split.is_coherent(seq)

    split = Split(0.2, 0.5, 0.3)
    assert split.is_ratio
    ex, mut, fix = split.apply(seq)
    assert ex == [1,2]
    assert mut == [3,4,5,6,7]
    assert fix == [8,9,10]

def test_split_conversion():
    split = Split(0.2, 0.5, 0.3)
    seq = [1,2,3,4,5,6,7,8,9,10]

    abs_split = split.to_abs(seq)
    assert abs_split == Split(2, 5, 3)

    ratio_split = abs_split.to_ratio(seq)
    assert ratio_split == split
