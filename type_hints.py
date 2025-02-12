from enum import Enum
from typing import List, Literal, Set, Tuple, TypeAlias

from aalpy.automata.Dfa import DfaState
from torch import Tensor

LabeledTensor: TypeAlias = Tuple[Tensor, int]

CategorySet: TypeAlias = Set[int]
CategorizedTensor = Tuple[Tensor, List[CategorySet]]

Dataset: TypeAlias = List[LabeledTensor]
CategorizedDataset: TypeAlias = List[CategorizedTensor]

GoodBadDataset: TypeAlias = Tuple[Dataset, Dataset]

TraceSplit: TypeAlias = Tuple[List[int], List[int], List[int]]
SplitTuple: TypeAlias = Tuple[float, float, float] | Tuple[int, int, int]
Trace: TypeAlias = List[Tensor | int]

PathInfo: TypeAlias = Tuple[
    int, int | float, int, DfaState, Tuple[DfaState, ...], Tuple[int, ...], int
]
PathsQueue: TypeAlias = List[PathInfo]


StrategyStr: TypeAlias = Literal[
    "genetic", "brute_force", "targeted", "genetic_categorized"
]



class RecDataset(Enum):
    ML_1M = "ml-1m"
    ML_100K = "ml-100k"


class RecModel(Enum):
    BERT4Rec = "BERT4Rec"
    GRU4Rec = "GRU4Rec"
    SASRec = "SASRec"
