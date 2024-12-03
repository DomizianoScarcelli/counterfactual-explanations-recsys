import time
from typing import Optional

import toml

from type_hints import RecDataset, RecModel

default_config_path = "configs/config.toml"

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
    def _parse_config(cls):
        """Load configuration and set class attributes."""
        if not cls._config_loaded:
            config = toml.load(cls._config_path)

            # Set parameters directly as class attributes
            cls.DEBUG = config["settings"]["debug"]
            cls.DETERMINISM = config["settings"]["determinism"]
            cls.MODEL = RecModel[config["settings"]["model"]]
            cls.DATASET = RecDataset[config["settings"]["dataset"]]
            cls.TRAIN_BATCH_SIZE = config["settings"]["train_batch_size"]
            cls.TEST_BATCH_SIZE = config["settings"]["test_batch_size"]

            cls.INCLUDE_SINK = config["automata"]["include_sink"]

            cls.GENERATIONS = config["evolution"]["generations"]
            cls.POP_SIZE = config["evolution"]["pop_size"]
            cls.HALLOFFAME_RATIO = config["evolution"]["halloffame_ratio"]
            cls.ALLOWED_MUTATIONS = config["evolution"]["allowed_mutations"]

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
    def fix(cls):
        """Make the config path non-reloadable."""
        cls._reloadable = False

    @classmethod
    def configs_dict(cls):
        return {
                "determinism": [ConfigParams.DETERMINISM],
                "model": [ConfigParams.MODEL.value],
                "dataset": [ConfigParams.DATASET.value],
                "pop_size": [ConfigParams.POP_SIZE],
                "generations": [ConfigParams.GENERATIONS],
                "halloffame_ratio": [ConfigParams.HALLOFFAME_RATIO],
                "allowed_mutations": [tuple(ConfigParams.ALLOWED_MUTATIONS)],
                "timestamp": [ConfigParams.TIMESTAMP]}

# Load default configs
ConfigParams()
