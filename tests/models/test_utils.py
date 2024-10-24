import re

import pytest
import torch

from models.utils import pad, pad_batch, trim

class TestTrimZeros:
    def test_trim_zero_valid(self):
        x = torch.tensor([1,2,0,4,5,6,7,8,-1,-1,-1,-1,-1])
        trimmed = trim(x)
        expected = torch.tensor([1,2,0,4,5,6,7,8])
        assert torch.all(trimmed == expected), f"Trimmed tensor is not equal to expected trimmed tensor" 

    def test_trim_zero_multidim(self):
        x = torch.tensor([1,2,3,4,5,6,7,8,-1,-1-1,-1,-1,-1]).unsqueeze(0)
        with pytest.raises(AssertionError, match=re.escape(f"Sequence must have a single dim, {x.shape}")):
            trim(x)

    def test_trim_zero_multiple_zeros(self):
        x = torch.tensor([1,2,-1, 3,4,5,6,7,8,-1,-1,-1,-1,-1])
        with pytest.raises(AssertionError, match=f"Sequence must use the character -1 only for padding!"):
            trim(x)

class TestPadZeros:
    def test_pad_zeros_valid(self):
        x = torch.tensor([1,2,3,4,5,6,7,8])
        trimmed = pad(x, len(x) + 6)
        expected = torch.tensor([1,2,3,4,5,6,7,8,-1,-1,-1,-1,-1,-1])
        assert torch.all(trimmed == expected), f"Padded tensor is not equal to expected padded tensor" 

    def test_fake_pad_zeros(self):
        x = torch.tensor([1,2,3,4,5,6,7,8])
        trimmed = pad(x, len(x))
        expected = torch.tensor([1,2,3,4,5,6,7,8])
        assert torch.all(trimmed == expected), f"Padded tensor is not equal to expected padded tensor" 

    def test_pad_zero_multidim(self):
        x = torch.tensor([1,2,3,4,5,6,7,8]).unsqueeze(0)
        with pytest.raises(AssertionError, match=re.escape(f"Sequence must have a single dim, {x.shape}")):
            pad(x, len(x) + 6)

    def test_pad_zero_multiple_zeros(self):
        x = torch.tensor([0,2,-1,4,5,6,7,8])
        with pytest.raises(AssertionError, match="Sequence must use the character -1 only for padding!" + r".*"):
            pad(x, len(x) + 6)

class TestPadZerosBatch:
    def test_pad_zeros_batch_valid(self):
        x = torch.tensor([1,2,3,4,5,6,7,8])
        x_1 = torch.tensor([1,2,3,4,5,6,7,8, 9, 10, 11])
        x_2 = torch.tensor([1,2])
        batch = [x, x_1, x_2]
        length = 12
        expected = torch.tensor([[1,2,3,4,5,6,7,8, -1, -1, -1, -1],
                                 [1,2,3,4,5,6,7,8, 9, 10, 11, -1],
                                 [1,2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
        padded = pad_batch(batch, length)
        assert torch.all(padded == expected), f"Padded tensor is not equal to expected padded tensor" 
