import warnings
from pathlib import Path

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import get_model, init_seed

from type_hints import RecDataset, RecModel

warnings.simplefilter(action="ignore", category=FutureWarning)

# Bert4Rec hyperparameters taken from https://github.com/asash/bert4rec_repro
parameter_dict = {
    "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
    "eval_step": 1,
    "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": "uni100"},
    # "loss_type": "BPR",
    # "train_neg_sample_args":{'distribution': 'uniform', 'sample_num': 100},
    "MAX_ITEM_LIST_LENGTH": 50,
    "train_neg_sample_args": None,
    "train_batch_size": 128,
    "hidden_dropout_prob": 0.2,
    "attn_dropout_prob": 0.2,
    "n_heads": 1,
}

config = Config(
    model=RecModel.BERT4Rec.value,
    dataset=RecDataset.ML_1M.value,
    config_dict=parameter_dict,
)

# Initialize logger and seed
# init_logger(config)
init_seed(config["seed"], config["reproducibility"])

# Load dataset and pre-trained model
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# Load a pre-trained model checkpoint
model = get_model(config["model"])(config, train_data.dataset).to(config["device"])

# Perform inference
trainer = Trainer(config, model)
latest_checkpoint = Path("saved/BERT4Rec_ml1m.pth")
trainer.resume_checkpoint(latest_checkpoint)
# NOTE: uncomment this to perform the training, otherwise just the evaluation part will be performed
# results = trainer.fit(train_data, show_progress=True)
results = trainer.evaluate(test_data, show_progress=True)
# results = trainer.evaluate(test_data, show_progress=True, model_file=latest_checkpoint)
print(f"Results are", results)
