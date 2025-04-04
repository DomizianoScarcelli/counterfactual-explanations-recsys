from utils.Cached import Cached
import pickle
from pathlib import Path
import json
import os
import time
from typing import List, Optional, TypedDict

import toml
import torch

from type_hints import RecDataset, RecModel

default_config_path = "config/configs/config.toml"


class DebugConfig(TypedDict):
    debug: int
    profile: bool


class SettingsConfig(TypedDict):
    device: str
    model: str
    dataset: str
    determinism: bool
    train_batch_size: int
    test_batch_size: int
    topk: List[int]


class GenerationConfig(TypedDict):
    targeted: bool
    categorized: bool
    similarity_threshold: float
    ignore_genetic_split: bool
    genetic_topk: int


class AutomataConfig(TypedDict):
    include_sink: bool


class MutationConfig(TypedDict):
    num_replaces: int
    num_additions: int
    num_deletions: int


class EvolutionConfig(TypedDict):
    generations: int
    target_cat: str
    pop_size: int
    halloffame_ratio: float
    fitness_alpha: float
    mutations: MutationConfig
    allowed_mutations: List[str]
    mut_prob: float
    crossover_prob: float


class ConfigDict(TypedDict):
    debug: DebugConfig
    generation: GenerationConfig
    settings: SettingsConfig
    automata: AutomataConfig
    evolution: EvolutionConfig


def deep_update(config: dict, override: dict):
    """
    Recursively updates the `config` dictionary with values from `override`.
    If a key in `override` points to a dictionary, it updates the corresponding
    dictionary in `config` recursively.

    Args:
        config (dict): The original configuration dictionary.
        override (dict): The dictionary with the values to update.

    Returns:
        dict: The updated configuration dictionary.
    """
    for key, value in override.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            deep_update(config[key], value)  # Recurse into the sub-dictionary
        else:
            config[key] = value  # Directly update the key in config
    return config


class ConfigParams:
    _instance = None  # Singleton instance
    _config_loaded = False  # Flag to ensure config is loaded only once
    _config_path = default_config_path  # Default path
    _reloadable = True  # Flag to track whether the path can be reloaded

    def __new__(cls, config_path: Optional[str] = None):
        """Ensure only one instance of ConfigParams is created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if config_path:
                cls._config_path = config_path  # Update path if provided
            cls._parse_config()
            print(f"Default config loaded from {cls._config_path}")
        return cls._instance

    @classmethod
    def _parse_config(cls, _dict: Optional[ConfigDict] = None):
        """Load configuration and set class attributes."""
        if not cls._config_loaded:
            config = toml.load(cls._config_path) if not _dict else _dict
            if config["debug"]["profile"]:
                print(f"!!!!!PROFILING ACTIVATED, PERFORMANCE MAY BE DEGRADRED!!!!!")
                os.environ["LINE_PROFILE"] = "1"
            else:
                os.environ["LINE_PROFILE"] = "0"

            # Set parameters directly as class attributes
            cls.DEBUG = config["debug"]["debug"]
            cls.DETERMINISM = config["settings"]["determinism"]
            cls.SEED = config["settings"]["seed"]
            cls.DEVICE = torch.device(config["settings"]["device"])
            cls.MODEL = RecModel[config["settings"]["model"]]
            cls.DATASET = RecDataset[config["settings"]["dataset"]]
            cls.TRAIN_BATCH_SIZE = config["settings"]["train_batch_size"]
            cls.TEST_BATCH_SIZE = config["settings"]["test_batch_size"]
            cls.TOPK = config["settings"]["topk"]

            cls.INCLUDE_SINK = config["automata"]["include_sink"]

            cls.TARGETED = config["generation"]["targeted"]
            cls.CATEGORIZED = config["generation"]["categorized"]

            cls._legacy_compute_stategy()
            cls.THRESHOLD = config["generation"]["similarity_threshold"]
            cls.IGNORE_GEN_SPLIT = config["generation"]["ignore_genetic_split"]
            cls.GENETIC_TOPK = config["generation"]["genetic_topk"]

            cls.GENERATIONS = config["evolution"]["generations"]
            cls.TARGET_CAT = config["evolution"]["target_cat"]

            def get_remapped_dataset(dataset: RecDataset):
                def load_pickle(path: Path):
                    with open(path, "rb") as f:
                        return pickle.load(f)

                if dataset in list(RecDataset):
                    path = Path(f"data/{cls.DATASET.value}-SequentialDataset.pth")
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Sequential dataset at {path} not found, make sure to generate it by running an InteractionGenerator with that dataset."
                        )
                else:
                    raise NotImplementedError(
                        f"get_category_map not implemented for dataset {dataset}"
                    )

                return Cached(path, load_fn=load_pickle).get_data()

            def token2id(dataset: RecDataset, token: str) -> int:
                """Maps external item tokens to internal ids."""
                if dataset in [RecDataset.ML_1M, RecDataset.ML_100K]:
                    field = "item_id"
                elif dataset == RecDataset.STEAM:
                    field = "product_id"
                else:
                    raise ValueError(
                        f"Dataset {dataset} not supported (supported datsets are {list(RecDataset)})"
                    )
                remapped_dataset = get_remapped_dataset(dataset)
                return int(remapped_dataset.token2id(field, tokens=token))

            if not cls.CATEGORIZED and cls.TARGET_CAT != False:
                cls.TARGET_CAT = token2id(
                    token=str(cls.TARGET_CAT), dataset=cls.DATASET
                )
            if isinstance(cls.TARGET_CAT, str) and not cls.CATEGORIZED:
                cls.TARGET_CAT = int(cls.TARGET_CAT)
            cls.POP_SIZE = config["evolution"]["pop_size"]
            cls.HALLOFFAME_RATIO = config["evolution"]["halloffame_ratio"]
            cls.ALLOWED_MUTATIONS = config["evolution"]["allowed_mutations"]
            cls.FITNESS_ALPHA = config["evolution"]["fitness_alpha"]
            cls.MUT_PROB = config["evolution"]["mut_prob"]
            cls.CROSSOVER_PROB = config["evolution"]["crossover_prob"]

            cls.NUM_REPLACES = config["evolution"]["mutations"]["num_replaces"]
            cls.NUM_ADDITIONS = config["evolution"]["mutations"]["num_additions"]
            cls.NUM_DELETIONS = config["evolution"]["mutations"]["num_deletions"]

            cls.TIMESTAMP = time.strftime("%a, %d %b %Y %H:%M:%S")

            item_id_field_mapping = {
                RecDataset.ML_100K: "item_id",
                RecDataset.ML_1M: "item_id",
                RecDataset.STEAM: "product_id",
            }

            cls.ITEM_ID_FIELD = item_id_field_mapping[cls.DATASET]
            cls.ITEM_ID_LIST_FIELD = f"{cls.ITEM_ID_FIELD}_list"

            cls._config_loaded = True  # Flag that config is loaded

    @classmethod
    def __getattr__(cls, name):
        """Lazy load config parameters if accessed before being initialized."""
        if hasattr(cls, name):
            return getattr(cls, name)
        else:
            raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")

    @classmethod
    def _legacy_compute_stategy(cls):
        # TODO: remove generation strategy, this is done to make legacy code work
        assert isinstance(cls.TARGETED, bool) and isinstance(cls.CATEGORIZED, bool)
        if cls.TARGETED and cls.CATEGORIZED:
            cls.GENERATION_STRATEGY = "targeted"
        elif not cls.TARGETED and cls.CATEGORIZED:
            cls.GENERATION_STRATEGY = "genetic_categorized"
        elif not cls.TARGETED and not cls.CATEGORIZED:
            cls.GENERATION_STRATEGY = "genetic"
        else:
            cls.GENERATION_STRATEGY = "targeted_uncategorized"
            # raise NotImplementedError(
            #     f"targeted: {cls.TARGETED} of type{type(cls.TARGETED)}, categorized: {cls.CATEGORIZED} of type {type(cls.CATEGORIZED)} still not implemented"
            # )

    @classmethod
    def reload(cls, path: Optional[str]):
        """Allow setting a custom config file path."""
        if not cls._reloadable:
            raise ValueError(
                "Config path is no longer reloadable because .fix() has been called."
            )

        new_path = path if path else default_config_path
        if new_path == cls._config_path:
            return
        cls._config_path = new_path
        cls._config_loaded = False  # Reset loaded flag to reload the config
        cls._parse_config()
        print(f"Config reloaded from {cls._config_path}")

    @classmethod
    def override_params(cls, override: ConfigDict):
        config = toml.load(
            cls._config_path if cls._config_path else default_config_path
        )
        config = deep_update(config, override)
        if cls.DEBUG >= 1:
            print(f"Overwriting parameters with the new config {override}")
        cls.reload_from_dict(_dict=config)  # type: ignore
        cls._legacy_compute_stategy()

    @classmethod
    def reload_from_dict(cls, _dict: ConfigDict):
        """Allow setting a custom config file path."""
        if not cls._reloadable:
            raise ValueError(
                "Config path is no longer reloadable because .fix() has been called."
            )

        cls._config_path = None
        cls._config_loaded = False  # Reset loaded flag to reload the config
        cls._parse_config(_dict=_dict)
        print(f"Config reloaded from provided dictionary")

    @classmethod
    def fix(cls):
        """Make the config path non-reloadable."""
        cls._reloadable = False

    @classmethod
    def get_default_config(cls) -> ConfigDict:
        return toml.load(default_config_path)  # type: ignore

    @classmethod
    def configs_dict(cls, length=1, pandas: bool = True, tostr: bool = False):
        conf = {
            "determinism": [ConfigParams.DETERMINISM] * length,
            "model": [ConfigParams.MODEL.value] * length,
            "dataset": [ConfigParams.DATASET.value] * length,
            "target_cat": [str(ConfigParams.TARGET_CAT)] * length,
            "pop_size": [ConfigParams.POP_SIZE] * length,
            "generations": [ConfigParams.GENERATIONS] * length,
            "halloffame_ratio": [ConfigParams.HALLOFFAME_RATIO] * length,
            "fitness_alpha": [ConfigParams.FITNESS_ALPHA] * length,
            "allowed_mutations": [tuple(ConfigParams.ALLOWED_MUTATIONS)] * length,
            "include_sink": [ConfigParams.INCLUDE_SINK] * length,
            "mut_prob": [ConfigParams.MUT_PROB] * length,
            "crossover_prob": [ConfigParams.CROSSOVER_PROB] * length,
            "genetic_topk": [ConfigParams.GENETIC_TOPK] * length,
            "mutation_params": [
                (
                    ConfigParams.NUM_REPLACES,
                    ConfigParams.NUM_ADDITIONS,
                    ConfigParams.NUM_DELETIONS,
                )
            ]
            * length,
            "generation_strategy": [ConfigParams.GENERATION_STRATEGY] * length,
            "ignore_genetic_split": [ConfigParams.IGNORE_GEN_SPLIT] * length,
            "jaccard_threshold": [ConfigParams.THRESHOLD] * length,
            "seed": [ConfigParams.SEED] * length,
            "timestamp": [ConfigParams.TIMESTAMP] * length,
        }
        if not pandas:
            conf = {key: value[0] for key, value in conf.items()}
        if tostr:
            conf = {key: str(value) for key, value in conf.items()}
        return conf

    @classmethod
    def print_config(cls, indent: Optional[int] = None):
        config_dict = cls.configs_dict()
        if indent:
            config_dict = json.dumps(config_dict, indent=indent)
        print(config_dict)


# Load default configs
ConfigParams()
