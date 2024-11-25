import toml
import time
from typing import Optional
from type_hints import RecDataset, RecModel

default_config_path = "configs/config.toml"

class ConfigParams:
    _instance = None  # Singleton instance
    _config_loaded = False  # Flag to ensure config is loaded only once
    _config_path = default_config_path  # Default path

    def __new__(cls, config_path: Optional[str] = None):
        """Ensure only one instance of ConfigParams is created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if config_path:
                cls._config_path = config_path  # Update path if provided
            cls._parse_config()
        return cls._instance

    @classmethod
    def _parse_config(cls):
        """Load configuration and set class attributes."""
        if not cls._config_loaded:
            print(f"Config generated from {cls._config_path}")
            config = toml.load(cls._config_path)

            # Set parameters directly as class attributes
            cls.DEBUG = config["settings"]["debug"]
            cls.DETERMINISM = config["settings"]["determinism"]
            cls.MODEL = RecModel[config["settings"]["model"]]
            cls.DATASET = RecDataset[config["settings"]["dataset"]]

            cls.GENERATIONS = config["evolution"]["generations"]
            cls.POP_SIZE = config["evolution"]["pop_size"]
            cls.HALLOFFAME_RATIO = config["evolution"]["halloffame_ratio"]
            cls.ALLOWED_MUTATIONS = config["evolution"]["allowed_mutations"]

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
    def reload(cls, path: str):
        """Allow setting a custom config file path."""
        cls._config_path = path
        cls._config_loaded = False  # Reset loaded flag to reload the config
        cls._parse_config()

# Load default configs
ConfigParams()

