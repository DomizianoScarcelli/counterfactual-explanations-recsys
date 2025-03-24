from recbole.utils import dataset_arguments
from config.constants import SUPPORTED_DATASETS
from pathlib import Path

import torch
from recbole.config import Config
from recbole.model.abstract_recommender import SequentialRecommender

from config.config import ConfigParams
from core.generation.dataset.utils import get_dataloaders
from core.models.extended_models.ExtendedBERT4Rec import ExtendedBERT4Rec
from core.models.extended_models.ExtendedGRU4Rec import ExtendedGRU4Rec
from core.models.extended_models.ExtendedSASRec import ExtendedSASRec
from type_hints import RecDataset, RecModel
from utils.utils import printd


def generate_model(config: Config) -> SequentialRecommender:
    """
    Creates the pytorch model from the config file.

    Args:
        config: the Config file, which contains the type of model that has to be created.
    Returns:
        The model.
    """
    train_data, _, _ = get_dataloaders(config)
    checkpoint_file = Path(
        f"saved_models/{ConfigParams.MODEL.value}_{ConfigParams.DATASET.value}.pth"
    )

    if config.model == RecModel.BERT4Rec.value:
        model = ExtendedBERT4Rec(config, train_data.dataset)
    elif config.model == RecModel.SASRec.value:
        model = ExtendedSASRec(config, train_data.dataset)
    elif config.model == RecModel.GRU4Rec.value:
        model = ExtendedGRU4Rec(config, train_data.dataset)
    else:
        raise ValueError(f"Model {config.model} not supported")
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=config["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
    return model.to(ConfigParams.DEVICE)


def get_config(
    dataset: RecDataset, model: RecModel, save_dataset: bool = True
) -> Config:
    printd(f"Loaded dataset: {dataset}", level=1)
    load_col = {"inter": ["user_id", "item_id", "timestamp"]}
    if ConfigParams.DATASET == RecDataset.STEAM:
        load_col = {"inter": ["user_id", "product_id", "timestamp"]}
    parameter_dict = {
        "ITEM_ID_FIELD": ConfigParams.ITEM_ID_FIELD,
        "USER_ID_FIELD": "user_id",
        "checkpoint_dir": "data/",
        "load_col": load_col,
        "train_neg_sample_args": None,
        "eval_batch_size": ConfigParams.TEST_BATCH_SIZE,
        "MAX_ITEM_LIST_LENGTH": 50,
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            # "mode": "uni100",
        },
        "save_dataset": save_dataset,
        "train_batch_size": ConfigParams.TRAIN_BATCH_SIZE,
        "device": ConfigParams.DEVICE,
        "seed": (
            ConfigParams.SEED if ConfigParams.SEED != 42 else 2020
        ),  # since in experiments with seed 42, the RecBole seed was left as default, we leave it as the default 2020 value when the seed is 42.
        # "n_heads": 1,
    }

    # print(f"[DEBUG] parameter dict is", parameter_dict)
    conf = Config(model=model.value, dataset=dataset.value, config_dict=parameter_dict)
    # print(f"[DEBUG] RecBole Config:", conf)
    return conf
