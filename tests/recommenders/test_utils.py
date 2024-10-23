import re

import pytest
import torch

from config import DATASET, MODEL
from genetic.utils import NumItems
from genetic.dataset.utils import get_sequence_from_interaction
from genetic.dataset.generate import interaction_generator
from recommenders.utils import pad_zero, pad_zero_batch, trim_zero
from recommenders.config_utils import get_config

class TestTrimZeros:
    def test_trim_zero_valid(self):
        x = torch.tensor([1,2,3,4,5,6,7,8,0,0,0,0,0,0])
        trimmed = trim_zero(x)
        expected = torch.tensor([1,2,3,4,5,6,7,8])
        assert torch.all(trimmed == expected), f"Trimmed tensor is not equal to expected trimmed tensor" 

    def test_trim_zero_multidim(self):
        x = torch.tensor([1,2,3,4,5,6,7,8,0,0,0,0,0,0]).unsqueeze(0)
        with pytest.raises(AssertionError, match=re.escape(f"Sequence must have a single dim, {x.shape}")):
            trim_zero(x)

    def test_trim_zero_multiple_zeros(self):
        x = torch.tensor([1,2,3,4,0,6,7,8,0,0,0,0,0,0])
        with pytest.raises(AssertionError, match=f"Sequence must use the character 0 only for padding!"):
            trim_zero(x)

class TestPadZeros:
    def test_pad_zeros_valid(self):
        x = torch.tensor([1,2,3,4,5,6,7,8])
        trimmed = pad_zero(x, len(x) + 6)
        expected = torch.tensor([1,2,3,4,5,6,7,8,0,0,0,0,0,0])
        assert torch.all(trimmed == expected), f"Padded tensor is not equal to expected padded tensor" 

    def test_fake_pad_zeros(self):
        x = torch.tensor([1,2,3,4,5,6,7,8])
        trimmed = pad_zero(x, len(x))
        expected = torch.tensor([1,2,3,4,5,6,7,8])
        assert torch.all(trimmed == expected), f"Padded tensor is not equal to expected padded tensor" 

    def test_pad_zero_multidim(self):
        x = torch.tensor([1,2,3,4,5,6,7,8]).unsqueeze(0)
        with pytest.raises(AssertionError, match=re.escape(f"Sequence must have a single dim, {x.shape}")):
            pad_zero(x, len(x) + 6)

    def test_pad_zero_multiple_zeros(self):
        x = torch.tensor([1,2,0,4,5,6,7,8])
        with pytest.raises(AssertionError, match=f"Sequence must not contain the character 0!"):
            pad_zero(x, len(x) + 6)

class TestPadZerosBatch:
    def test_pad_zeros_batch_valid(self):
        x = torch.tensor([1,2,3,4,5,6,7,8])
        x_1 = torch.tensor([1,2,3,4,5,6,7,8, 9, 10, 11])
        x_2 = torch.tensor([1,2])
        batch = [x, x_1, x_2]
        length = 12
        expected = torch.tensor([[1,2,3,4,5,6,7,8, 0, 0, 0, 0],
                                 [1,2,3,4,5,6,7,8, 9, 10, 11, 0],
                                 [1,2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        padded = pad_zero_batch(batch, length)
        assert torch.all(padded == expected), f"Padded tensor is not equal to expected padded tensor" 

@pytest.mark.skip()
class TestGenerators:
    def test_interaction_generator(self):
        config = get_config(model=MODEL, dataset=DATASET)
        interactions = interaction_generator(config)
        items = set()
        for interaction in interactions:
            seq = get_sequence_from_interaction(interaction).squeeze(0).tolist()
            for i in seq:
                items.add(i)
        assert min(items) == 0 and max(items) == NumItems.ML_1M
            
