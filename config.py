import json
import os
import time
from typing import List, Optional, TypedDict

import toml

from type_hints import RecDataset, RecModel

default_config_path = "configs/config.toml"

class DebugConfig(TypedDict):
    debug: int
    profile: bool
    
class SettingsConfig(TypedDict):
    model: str
    dataset: str
    determinism: bool
    train_batch_size: int
    test_batch_size: int

class GenerationConfig(TypedDict):
    strategy: str

class AutomataConfig(TypedDict):
    include_sink: bool

class MutationConfig(TypedDict):
    num_replaces: int
    num_additions: int
    num_deletions: int

class EvolutionConfig(TypedDict):
    generations: int
    pop_size: int
    halloffame_ratio: float
    fitness_alpha: float
    mutations: MutationConfig
    allowed_mutations: List[str]

class ConfigDict(TypedDict):
    debug: DebugConfig
    generation: GenerationConfig
    settings: SettingsConfig
    automata: AutomataConfig
    evolution: EvolutionConfig

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
    def _parse_config(cls, _dict: Optional[ConfigDict]=None):
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
            cls.MODEL = RecModel[config["settings"]["model"]]
            cls.DATASET = RecDataset[config["settings"]["dataset"]]
            cls.TRAIN_BATCH_SIZE = config["settings"]["train_batch_size"]
            cls.TEST_BATCH_SIZE = config["settings"]["test_batch_size"]

            cls.INCLUDE_SINK = config["automata"]["include_sink"]

            cls.GENERATION_STRATEGY = config["generation"]["strategy"]

            cls.GENERATIONS = config["evolution"]["generations"]
            cls.POP_SIZE = config["evolution"]["pop_size"]
            cls.HALLOFFAME_RATIO = config["evolution"]["halloffame_ratio"]
            cls.ALLOWED_MUTATIONS = config["evolution"]["allowed_mutations"]
            cls.FITNESS_ALPHA = config["evolution"]["fitness_alpha"]

            cls.NUM_REPLACES = config["evolution"]["mutations"]["num_replaces"]
            cls.NUM_ADDITIONS = config["evolution"]["mutations"]["num_additions"]
            cls.NUM_DELETIONS = config["evolution"]["mutations"]["num_deletions"]

            cls.TIMESTAMP = time.strftime("%a, %d %b %Y %H:%M:%S")

            cls._config_loaded = True  # Flag that config is loaded

    @classmethod
    def __getattr__(cls, name):
        """Lazy load config parameters if accessed before being initialized."""
        if hasattr(cls, name):
            return getattr(cls, name)
        else:
            raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")

    @classmethod
    def reload(cls, path: Optional[str]):
        """Allow setting a custom config file path."""
        if not cls._reloadable:
            raise ValueError("Config path is no longer reloadable because .fix() has been called.")
        
        new_path = path if path else default_config_path
        if new_path == cls._config_path:
            return
        cls._config_path = new_path
        cls._config_loaded = False  # Reset loaded flag to reload the config
        cls._parse_config()
        print(f"Config reloaded from {cls._config_path}")

    @classmethod
    def reload_from_dict(cls, _dict: ConfigDict):
        """Allow setting a custom config file path."""
        if not cls._reloadable:
            raise ValueError("Config path is no longer reloadable because .fix() has been called.")
        
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
        return toml.load(default_config_path) #type: ignore

    @classmethod
    def configs_dict(cls):
        return {
                "determinism": [ConfigParams.DETERMINISM],
                "model": [ConfigParams.MODEL.value],
                "dataset": [ConfigParams.DATASET.value],
                "pop_size": [ConfigParams.POP_SIZE],
                "generations": [ConfigParams.GENERATIONS],
                "halloffame_ratio": [ConfigParams.HALLOFFAME_RATIO],
                "fitness_alpha": [ConfigParams.FITNESS_ALPHA],
                "allowed_mutations": [tuple(ConfigParams.ALLOWED_MUTATIONS)],
                "include_sink": [ConfigParams.INCLUDE_SINK],
                "mutation_params": [(ConfigParams.NUM_REPLACES, ConfigParams.NUM_ADDITIONS, ConfigParams.NUM_DELETIONS)],
                "generation_strategy": [ConfigParams.GENERATION_STRATEGY],
                "timestamp": [ConfigParams.TIMESTAMP]}

    @classmethod
    def print_config(cls, indent: Optional[int]=None):
        config_dict = cls.configs_dict()
        if indent:
            config_dict = json.dumps(config_dict, indent=indent)
        print(config_dict)

# Load default configs
ConfigParams()
