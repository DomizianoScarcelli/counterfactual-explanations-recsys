from type_hints import RecDataset, RecModel
import toml

config = toml.load("configs/config.toml")

DEBUG = config["settings"]["debug"]
MODEL = RecModel[config["settings"]["model"]]
DATASET = RecDataset[config["settings"]["dataset"]]

GENERATIONS = config["evolution"]["generations"]
POP_SIZE = config["evolution"]["pop_size"]
HALLOFFAME_RATIO = config["evolution"]["halloffame_ratio"]
