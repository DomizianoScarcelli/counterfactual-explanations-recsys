from torch import Tensor
from typing import List, Tuple, TypeAlias
from enum import Enum

LabeledTensor: TypeAlias = Tuple[Tensor, int]
Dataset: TypeAlias = List[LabeledTensor]

class RecDataset(Enum):
    ML_1M = "ml-1m"

class RecModel(Enum):
    BERT4Rec = "BERT4Rec"
