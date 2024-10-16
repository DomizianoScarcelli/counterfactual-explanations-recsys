from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import get_model

parameter_dict = {
        'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
        'train_neg_sample_args': None
        }

config = Config(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict)

# Initialize logger and seed
# init_logger(config)
init_seed(config['seed'], config['reproducibility'])

# Load dataset and pre-trained model
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# Load a pre-trained model checkpoint
model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

# Perform inference
trainer = Trainer(config, model)
latest_checkpoint = "saved/Bert4Rec_ml1m.pth"
trainer.resume_checkpoint(latest_checkpoint)
# results = trainer.fit(train_data, show_progress=True)
results = trainer.evaluate(test_data, show_progress=True, model_file=latest_checkpoint)
print(results)
