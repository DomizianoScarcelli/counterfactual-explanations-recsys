from utils.generators import InteractionGenerator
from config.constants import SUPPORTED_DATASETS
import json
from pathlib import Path
import fire

import pandas as pd
from tqdm import tqdm

from config.config import ConfigParams
from core.generation.utils import token2id
from core.models.config_utils import get_config
from type_hints import RecDataset


def main(dataset):
    generate_dataset_pth(dataset)
    create_category_mapping(dataset)


def generate_dataset_pth(dataset):
    path = Path(f"data/{dataset}-SequentialDataset.pth")
    if path.exists():
        print(f"{path.name} already exists, skipping pth generation")
        return
    print(f"{path.name} doesn't exists, generating...")
    confg = get_config(model=ConfigParams.MODEL, dataset=dataset)
    ints = InteractionGenerator(confg)
    next(ints)
    assert (
        path.exists()
    ), f"Error in generating the dataset pth, {path.name} doesn't exists"
    print(f"{path.name} generated")


def steam_download():
    pass


def get_dataset(dataset: RecDataset):
    dataset_path = Path(f"dataset/{dataset.value}")
    get_config(dataset=dataset, model=ConfigParams.MODEL, save_dataset=True)
    if not dataset_path.exists():
        # Generate the dataset
        if dataset == RecDataset.STEAM:
            steam_download()
        else:
            raise ValueError(
                f"Dataset {dataset.value} not found, make sure to download it following the instructions at https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools/usage and putting the unzipped and processed directory in the dataset/"
            )
    return dataset_path


def create_category_mapping(dataset):
    if dataset in ["ml-100", "ml-1m"]:
        movielens_mapping()
    elif dataset in ["steam"]:
        steam_mapping()
    else:
        raise ValueError(
            f"Dataset {dataset} not supported, supported datasets are: {[x.value for x in SUPPORTED_DATASETS]}"
        )


def steam_mapping():
    dataset = RecDataset.STEAM
    dataset_path = get_dataset(dataset)

    inter_info_path = dataset_path / Path(f"{dataset.value}.inter")
    item_info_path = dataset_path / Path(f"{dataset.value}.item")
    inter_df = pd.read_csv(inter_info_path, delimiter="\t")
    item_df = pd.read_csv(item_info_path, delimiter="\t")[
        ["id:token", "genres:token_seq"]
    ]
    genres = dict(zip(item_df["id:token"], item_df["genres:token_seq"]))
    category_map = {}
    for _, row in tqdm(inter_df.iterrows(), total=inter_df.shape[0]):
        product_id = str(row["product_id:token"])
        try:
            id = token2id(dataset, product_id)
            category_map[id] = genres[int(product_id)]
        except ValueError:
            print(f"Value error on {product_id}")
            continue

    print("[DEBUG] category map is", category_map)
    with open(f"data/category_map_{dataset.value}.json", "w") as f:
        json.dump(category_map, f, indent=2)


def movielens_mapping():
    """
    This script processes the MovieLens 1M dataset to create a mapping between internal item IDs and their associated
    categories (genres). The resulting mapping is saved_models as a JSON file for further use.

    The json file is used by `generation.utils.get_category_map()`
    """
    dataset = ConfigParams.DATASET
    dataset_path = get_dataset(dataset)

    item_info_path = dataset_path / Path(f"{ConfigParams.DATASET.value}.item")
    df = pd.read_csv(item_info_path, delimiter="\t")

    # maps interals ids to categories
    category_map = {}
    for _, row in tqdm(df.iterrows()):
        if dataset in [RecDataset.ML_1M, RecDataset.ML_100K]:
            token = str(row["item_id:token"])
        elif dataset in [RecDataset.STEAM]:
            token = str(row["product_id:token"])
        else:
            raise ValueError(
                f"Dataset {dataset} not supported (supported datsets are {SUPPORTED_DATASETS})"
            )
        try:
            id = token2id(dataset, token)
            if dataset == RecDataset.ML_1M:
                genre_key = "genre:token_seq"
            elif dataset == RecDataset.ML_100K:
                genre_key = "class:token_seq"
            else:
                raise ValueError(f"Dataset {dataset} is not supported")
            category_map[id] = row[genre_key].split(" ")  # type: ignore
        except ValueError:
            print(f"Value error on {token}")
            continue

    with open(f"data/category_map_{ConfigParams.DATASET.value}.json", "w") as f:
        json.dump(category_map, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
