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

def test_split_double_inference():
    split_end = Split(None, None, 1)
    split_middle = Split(10, None, None)
    split_start = Split(None, 11, None)

    seq = list(range(1, 21))

    inf_split_end = split_end.parse_nan(seq)
    inf_split_middle = split_middle.parse_nan(seq)
    inf_split_start = split_start.parse_nan(seq)

    assert inf_split_end in [Split(10, 9, 1), Split(9, 10, 1)]
    assert inf_split_middle == Split(10, 5, 5)
    assert inf_split_start in [Split(5, 11, 4), Split(4, 11, 5)]

def test_split_single_inference():
    split_end = Split(None, 10, 1)
    split_middle = Split(10, 10, None)
    split_start = Split(3, 11, None)

    seq = list(range(1, 21))

    inf_split_end = split_end.parse_nan(seq)
    inf_split_middle = split_middle.parse_nan(seq)
    inf_split_start = split_start.parse_nan(seq)

    assert inf_split_end == Split(9, 10, 1)
    assert inf_split_middle == Split(10, 10, 0)
    assert inf_split_start == Split(3, 11, 6)
