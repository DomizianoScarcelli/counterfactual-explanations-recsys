from enum import Enum
from typing import List, Tuple, TypeAlias

from torch import Tensor

LabeledTensor: TypeAlias = Tuple[Tensor, int]
Dataset: TypeAlias = List[LabeledTensor]

class RecDataset(Enum):
    ML_1M = "ml-1m"

class RecModel(Enum):
    BERT4Rec = "BERT4Rec"
