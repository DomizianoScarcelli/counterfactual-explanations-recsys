import json
import os

import pandas as pd

from config import ConfigParams
from generation.utils import token2id
from models.config_utils import get_config
from type_hints import RecDataset


def create_category_mapping():
    """
    This script processes the MovieLens 1M dataset to create a mapping between internal item IDs and their associated
    categories (genres). The resulting mapping is saved as a JSON file for further use.

    The json file is used by `generation.utils.get_category_map()`
    """
    dataset_path = f"dataset/{ConfigParams.DATASET.value}"
    get_config(
        dataset=ConfigParams.DATASET, model=ConfigParams.MODEL, save_dataset=True
    )
    if not os.path.exists(dataset_path):
        # Generate the dataset
        print(f"")
        raise ValueError(
            f"Dataset {ConfigParams.DATASET.value} not found, make sure to download the .zip from https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI and unzip it in the dataset/ directory"
        )

    item_info_path = os.path.join(dataset_path, f"{ConfigParams.DATASET.value}.item")
    df = pd.read_csv(item_info_path, delimiter="\t")

    # maps interals ids to categories
    category_map = {}
    for _, row in df.iterrows():
        token = str(row["item_id:token"])
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
