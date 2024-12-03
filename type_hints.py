from enum import Enum
from typing import List, Tuple, TypeAlias
from aalpy.automata.Dfa import DfaState
from torch import Tensor

LabeledTensor: TypeAlias = Tuple[Tensor, int]
Dataset: TypeAlias = List[LabeledTensor]
GoodBadDataset: TypeAlias = Tuple[Dataset, Dataset]
TraceSplit: TypeAlias = Tuple[List[int], List[int], List[int]]
SplitTuple: TypeAlias = Tuple[float, float, float] | Tuple[int, int, int]
Trace: TypeAlias = List[Tensor|int]
PathInfo: TypeAlias = Tuple[int, int | float, int, DfaState, Tuple[DfaState, ...], Tuple[int, ...], int]
PathsQueue: TypeAlias = List[PathInfo]

class RecDataset(Enum):
    ML_1M = "ml-1m"

class RecModel(Enum):
    BERT4Rec = "BERT4Rec"
    SASRec = "SASRec"

