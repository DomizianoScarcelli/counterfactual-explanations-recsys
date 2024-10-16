import re

import pytest
import torch

from recommenders.utils import pad_zero, trim_zero


def test_trim_zero_valid():
    x = torch.tensor([1,2,3,4,5,6,7,8,0,0,0,0,0,0])
    trimmed = trim_zero(x)
    expected = torch.tensor([1,2,3,4,5,6,7,8])
    assert torch.all(trimmed == expected), f"Trimmed tensor is not equal to expected trimmed tensor" 

def test_trim_zero_multidim():
    x = torch.tensor([1,2,3,4,5,6,7,8,0,0,0,0,0,0]).unsqueeze(0)
    with pytest.raises(AssertionError, match=re.escape(f"Sequence must have a single dim, {x.shape}")):
        trim_zero(x)

def test_trim_zero_multiple_zeros():
    x = torch.tensor([1,2,3,4,0,6,7,8,0,0,0,0,0,0])
    with pytest.raises(AssertionError, match=f"Sequence must use the character 0 only for padding!"):
        trim_zero(x)

def test_pad_zeros_valid():
    x = torch.tensor([1,2,3,4,5,6,7,8])
    trimmed = pad_zero(x, len(x) + 6)
    expected = torch.tensor([1,2,3,4,5,6,7,8,0,0,0,0,0,0])
    assert torch.all(trimmed == expected), f"Padded tensor is not equal to expected padded tensor" 

def test_fake_pad_zeros():
    x = torch.tensor([1,2,3,4,5,6,7,8])
    trimmed = pad_zero(x, len(x))
    expected = torch.tensor([1,2,3,4,5,6,7,8])
    assert torch.all(trimmed == expected), f"Padded tensor is not equal to expected padded tensor" 

def test_pad_zero_multidim():
    x = torch.tensor([1,2,3,4,5,6,7,8]).unsqueeze(0)
    with pytest.raises(AssertionError, match=re.escape(f"Sequence must have a single dim, {x.shape}")):
        pad_zero(x, len(x) + 6)

def test_pad_zero_multiple_zeros():
    x = torch.tensor([1,2,0,4,5,6,7,8])
    with pytest.raises(AssertionError, match=f"Sequence must not contain the character 0!"):
        pad_zero(x, len(x) + 6)
