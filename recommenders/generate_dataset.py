from recbole.utils import init_seed, get_model
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import pickle
from typing import Generator, Tuple
from deap_generator import GeneticGenerationStrategy
from recommenders.model_funcs import model_predict
from type_hints import Dataset, RecDataset, RecModel


def generate_counterfactual_dataset(interaction: Interaction, model: SequentialRecommender) -> Tuple[Dataset,Dataset]:
    """
    Generates the dataset of good and bad points from a sequence in the
    Interaction, using the model as a black box oracle. The dataset can be used
    to train an automata that accepts good sequences and rejects bad sequences
    using the functions in `automata_learning.py`


    Args:
        interaction: the Interaction object containing the sequence for which a
        counterfactual is needed.
        model: the black box model that is used as an oracle to get the
        probability distribution or label of the sequence.
    Returns:
        The dataset in the form of a tuple (good_points, bad_points), where
        good_points and bad_points are lists of LabeledTensors.
    """
    sequence = get_sequence_from_interaction(interaction)
    assert sequence.size(0) == 1, f"Only batch size of 1 is supported, sequence shape is: {sequence.shape}"
    # user_id = interaction.interaction["user_id"][0].item()

    #Trim zeros
    #TODO: don't know if this is working
    sequence = sequence[:torch.nonzero(sequence, as_tuple=False).max().item() + 1] if sequence.nonzero().size(0) > 0 else torch.tensor([])
    sequence = sequence.squeeze(0)
    assert len(sequence.shape) == 1, f"Sequence dim must be 1: {sequence.shape}"
    good_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                      predictor=lambda x: model_predict(seq=x,
                                                                    interaction=interaction,
                                                                    model=model,
                                                                    prob=True),
                                                      pop_size=1000,
                                                      good_examples=True,
                                                      generations=10)
    good_examples = good_genetic_strategy.generate()
    good_examples = good_genetic_strategy.postprocess(good_examples)
    bad_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                     predictor=lambda x: model_predict(seq=x,
                                                                   interaction=interaction,
                                                                   model=model,
                                                                   prob=True),
                                                     pop_size=1000,
                                                     good_examples=False,
                                                     generations=10)
    bad_examples = bad_genetic_strategy.generate()
    bad_examples = bad_genetic_strategy.postprocess(bad_examples)
    
    return make_deterministic((good_examples, bad_examples))

def save_dataset(dataset: Tuple[Dataset, Dataset], save_path: str):
    """
    Saves the dataset as a .pickle file

    Args:
        dataset: The dataset that has to be saved.
        save_path: The save path
    """
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

def load_dataset(load_path: str) -> Tuple[Dataset, Dataset]:
    """
    Loads the dataset from disk.

    Args:
        load_path: The path from which to load the dataset
    Returns:
        The dataset (good_points, bad_points).
    """
    with open(load_path, "rb") as f:
        return pickle.load(f)


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates the train, val, test dataloaders from the dataset in the config file.

    Args:
        config: the Config file, which contains the dataset used to create the pytorch dataloaders.
    Returns:
        The train, val, test torch Dataloaders.
    """
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return train_data, valid_data, test_data 

def generate_model(config: Config) -> SequentialRecommender:
    """
    Creates the pytorch model from the config file.

    Args:
        config: the Config file, which contains the type of model that has to be created.
    Returns:
        The model.
    """
    train_data, _, _= get_dataloaders(config)
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    checkpoint_file = "saved/Bert4Rec_ml1m.pth"
    checkpoint = torch.load(checkpoint_file, map_location=config['device'])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    return model

def get_config(dataset: RecDataset, model: RecModel) -> Config:
    parameter_dict_ml1m = {
            'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
            'train_neg_sample_args': None,
            "eval_batch_size": 1}
    return Config(model=model.value, dataset=dataset.value, config_dict=parameter_dict_ml1m)

def get_sequence_from_interaction(interaction: Interaction) -> Tensor:
    sequence = interaction.interaction["item_id_list"] 
    print(f"[generate_dataset.get_sequence_from_interaction] sequence is {sequence}")
    return sequence

def interaction_generator(config: Config) -> Generator[Interaction, None, None]:
    # raw_interactions = interaction.interaction
    # user_ids = raw_interactions["user_id"]
    # item_ids = raw_interactions["item_id"]
    # item_matrix = raw_interactions["item_id_list"]
    # item_length = raw_interactions["item_length"]
    _, _, test_data = get_dataloaders(config)
    assert test_data is not None, "Test data is None"
    for data in test_data:
        interaction = data[0]
        yield interaction

def dataset_generator(config: Config) -> Generator[Tuple[Dataset, Dataset], None, None]:
    interactions = interaction_generator(config)
    model = generate_model(config)
    for interaction in interactions:
        good_examples, bad_examples = generate_counterfactual_dataset(interaction, model)
        
        yield (good_examples, bad_examples)
        break #generate just one dataset

def make_deterministic(dataset: Tuple[Dataset, Dataset]) -> Tuple[Dataset, Dataset]:
    g, b = dataset
    new_g, new_b = [], []
    ids = set()
    for p, l in g:
        if tuple(p.tolist()) in ids:
            continue
        new_g.append((p,l))
        ids.add(tuple(p.tolist()))
    for p, l in b:
        if tuple(p.tolist()) in ids:
            continue
        new_b.append((p,l))
        ids.add(tuple(p.tolist()))
    dataset = (new_g, new_b)
    return dataset

if __name__ == "__main__":
    config = get_config(model=RecModel.BERT4Rec, dataset=RecDataset.ML_1M)
    datasets = dataset_generator(config)
    for dataset in datasets:
        dataset_save_path = "saved/counterfactual_dataset.pickle"
        save_dataset(dataset, save_path=dataset_save_path)
        break




