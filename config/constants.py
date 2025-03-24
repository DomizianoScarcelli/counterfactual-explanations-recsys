import json
from enum import Enum
from pathlib import Path
from typing import Dict

from config.config import ConfigParams
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, EmptyDatasetError,
                        NoTargetStatesError, SplitNotCoherent)
from type_hints import RecDataset


class InputLength(Enum):
    Bert4Rec = (10, 50)


# The minimum and maximum length that a sequence may be during the generation generation
MIN_LENGTH, MAX_LENGTH = InputLength.Bert4Rec.value
PADDING_CHAR = -1


def get_cat2id() -> Dict[str, int]:
    dataset = ConfigParams.DATASET
    if dataset in [RecDataset.ML_100K, RecDataset.ML_1M]:
        return cat2id_movielens
    elif dataset == RecDataset.STEAM:
        return cat2id_steam()
    else:
        raise ValueError(f"Dataset {dataset.value} is not supported")


def get_id2cat():
    cat2id = get_cat2id()
    return {value: key for key, value in cat2id.items()}


def cat2id_steam():
    category_map_path = Path(f"data/category_map_{RecDataset.STEAM.value}.json")
    cat2id_path = Path(f"data/cat2id_{RecDataset.STEAM.value}.json")
    if cat2id_path.exists():
        with open(cat2id_path, "r") as f:
            return json.load(f)
    print(f"{cat2id_path.name} doesn't exists, generating it from {category_map_path}")
    with open(category_map_path, "r") as f:
        category_map = json.load(f)
    categories = set()
    for value in category_map.values():
        for cat in value:
            categories.add(cat)
    result = {cat: id for id, cat in enumerate(categories)}
    with open(cat2id_path, "w") as f:
        json.dump(result, f)
    return result


cat2id_movielens = {
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

cat2id = get_cat2id()
id2cat = get_id2cat()

# cat2id = id2cat= None

SUPPORTED_DATASETS = list(RecDataset)

error_messages = {
    DfaNotAccepting: "DfaNotAccepting",
    DfaNotRejecting: "DfaNotRejecting",
    NoTargetStatesError: "NoTargetStatesError",
    CounterfactualNotFound: "CounterfactualNotFound",
    SplitNotCoherent: "SplitNotCoherent",
    EmptyDatasetError: "EmptyDatasetError",
    KeyError: "KeyError",
}

if __name__ == "__main__":
    print(cat2id)
