import toml
from type_hints import RecDataset, RecModel
from typing import List

config = toml.load("configs/config.toml")

DEBUG = config["settings"]["debug"]
DETERMINISM = config["settings"]["determinism"]
MODEL = RecModel[config["settings"]["model"]]
DATASET = RecDataset[config["settings"]["dataset"]]

GENERATIONS = config["evolution"]["generations"]
POP_SIZE = config["evolution"]["pop_size"]
HALLOFFAME_RATIO = config["evolution"]["halloffame_ratio"]
ALLOWED_MUTATIONS = config["evolution"]["allowed_mutations"]


