import torch
from recbole.config import Config
from recbole.model.abstract_recommender import SequentialRecommender

from config import ConfigParams
from genetic.dataset.utils import get_dataloaders
from models.extended_models.ExtendedBERT4Rec import ExtendedBERT4Rec
from models.extended_models.ExtendedSASRec import ExtendedSASRec
from type_hints import RecDataset, RecModel


def generate_model(config: Config) -> SequentialRecommender:
    """
    Creates the pytorch model from the config file.

    Args:
        config: the Config file, which contains the type of model that has to be created.
    Returns:
        The model.
    """
    train_data, _, _ = get_dataloaders(config)
    checkpoint_map = {
        RecModel.BERT4Rec.value: "saved/Bert4Rec_ml1m.pth",
        RecModel.SASRec.value: "saved/SASRec_ml1m.pth",
    }

    if config.model == RecModel.BERT4Rec.value:
        model = ExtendedBERT4Rec(config, train_data.dataset)
    elif config.model == RecModel.SASRec.value:
        model = ExtendedSASRec(config, train_data.dataset)
    else:
        raise ValueError(f"Model {config.model} not supported")
    checkpoint_file = checkpoint_map[config.model]
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=config["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
    return model


def get_config(dataset: RecDataset, model: RecModel) -> Config:
    parameter_dict_ml1m = {
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "train_neg_sample_args": None,
        "eval_batch_size": ConfigParams.TEST_BATCH_SIZE,
        "MAX_ITEM_LIST_LENGTH": 50,
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            # "mode": "uni100",
        },
        "save_dataset": True,
        "train_batch_size": ConfigParams.TRAIN_BATCH_SIZE,
        # "n_heads": 1,
    }
    return Config(
        model=model.value, dataset=dataset.value, config_dict=parameter_dict_ml1m
    )
