from config.constants import SUPPORTED_DATASETS
import json
from pathlib import Path

import pandas as pd

from config.config import ConfigParams
from core.generation.utils import token2id
from core.models.config_utils import get_config
from type_hints import RecDataset


def create_category_mapping():
    """
    This script processes the MovieLens 1M dataset to create a mapping between internal item IDs and their associated
    categories (genres). The resulting mapping is saved_models as a JSON file for further use.

    The json file is used by `generation.utils.get_category_map()`
    """
    dataset = ConfigParams.DATASET
    dataset_path = Path(f"dataset/{dataset.value}")
    get_config(dataset=dataset, model=ConfigParams.MODEL, save_dataset=True)
    if not dataset_path.exists():
        # Generate the dataset
        raise ValueError(
            f"Dataset {dataset.value} not found, make sure to download it following the instructions at https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools/usage and putting the unzipped and processed directory in the dataset/"
        )

    item_info_path = dataset_path / Path(f"{ConfigParams.DATASET.value}.item")
    df = pd.read_csv(item_info_path, delimiter="\t")

    # maps interals ids to categories
    category_map = {}
    for _, row in df.iterrows():
        if dataset in [RecDataset.ML_1M, RecDataset.ML_100K]:
            token = str(row["item_id:token"])
        elif dataset in [RecDataset.LASTFM]:
            token = str(row["artist_id:token"])
        else:
            raise ValueError(
                f"Dataset {dataset} not supported (supported datsets are {SUPPORTED_DATASETS})"
            )
        try:
            id = token2id(ConfigParams.DATASET, token)
            if ConfigParams.DATASET == RecDataset.ML_1M:
                genre_key = "genre:token_seq"
            elif ConfigParams.DATASET == RecDataset.ML_100K:
                genre_key = "class:token_seq"
            else:
                raise ValueError(f"Dataset {ConfigParams.DATASET} is not supported")
            category_map[id] = row[genre_key].split(" ")  # type: ignore
        except ValueError:
            print(f"Value error on {token}")
            continue

    with open(f"data/category_map_{ConfigParams.DATASET.value}.json", "w") as f:
        json.dump(category_map, f, indent=2)


if __name__ == "__main__":
    create_category_mapping()
