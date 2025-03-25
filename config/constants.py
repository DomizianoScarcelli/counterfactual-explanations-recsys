from enum import Enum

from exceptions import (
    CounterfactualNotFound,
    DfaNotAccepting,
    DfaNotRejecting,
    EmptyDatasetError,
    NoTargetStatesError,
    SplitNotCoherent,
)
from type_hints import RecDataset


class InputLength(Enum):
    Bert4Rec = (10, 50)


# The minimum and maximum length that a sequence may be during the generation generation
MIN_LENGTH, MAX_LENGTH = InputLength.Bert4Rec.value
PADDING_CHAR = -1


cat2id = {
    "Action": 0,
    "Adventure": 1,
    "Animation": 2,
    "Children's": 3,
    "Comedy": 3,
    "Crime": 5,
    "Documentary": 6,
    "Drama": 7,
    "Fantasy": 8,
    "Film-Noir": 9,
    "Horror": 10,
    "Musical": 11,
    "Mystery": 12,
    "Romance": 13,
    "Sci-Fi": 14,
    "Thriller": 15,
    "War": 16,
    "Western": 17,
    "unknown": 18,  
}

id2cat = {value: key for key, value in cat2id.items()}

SUPPORTED_DATASETS = [RecDataset.ML_1M, RecDataset.ML_100K]

error_messages = {
    DfaNotAccepting: "DfaNotAccepting",
    DfaNotRejecting: "DfaNotRejecting",
    NoTargetStatesError: "NoTargetStatesError",
    CounterfactualNotFound: "CounterfactualNotFound",
    SplitNotCoherent: "SplitNotCoherent",
    EmptyDatasetError: "EmptyDatasetError",
    KeyError: "KeyError",
}
