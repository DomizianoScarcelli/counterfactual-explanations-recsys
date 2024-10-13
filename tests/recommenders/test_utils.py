import torch
from recommenders.utils import trim_zero

def test_trim_zero():
    x = torch.tensor([1,2,3,4,5,6,7,8,0,0,0,0,0,0])
    trimmed = trim_zero(x)
    expected = torch.tensor([1,2,3,4,5,6,7,8])
    assert torch.all(trimmed == expected), f"Trimmed tensor is not equal to expected trimmed tensor" 
