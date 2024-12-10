import json
import os

import pandas as pd

from genetic.utils import token2id
from type_hints import RecDataset


def create_category_mapping():
    """
    This script processes the MovieLens 1M dataset to create a mapping between internal item IDs and their associated
    categories (genres). The resulting mapping is saved as a JSON file for further use.

    The json file is used by `genetic.utils.get_category_map()`
    """
    dataset_path = "dataset/ml-1m"
    item_info_path = os.path.join(dataset_path, "ml-1m.item")

    df = pd.read_csv(item_info_path, delimiter="\t")

    # maps interals ids to categories
    category_map = {}
    for _, row in df.iterrows():
        token = str(row["item_id:token"])
        try:
            id = token2id(RecDataset.ML_1M, token)
            category_map[id] = row["genre:token_seq"].split(" ")
        except ValueError:
            continue

    with open("data/category_map.json", "w") as f:
        json.dump(category_map, f, indent=2)


if __name__ == "__main__":
    create_category_mapping()
