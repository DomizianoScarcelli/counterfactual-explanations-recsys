import os
from statistics import mean
from typing import Optional

import fire
import pandas as pd
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from config import ConfigParams
from constants import MAX_LENGTH
from genetic.utils import NumItems
from models.config_utils import generate_model, get_config
from models.utils import pad_batch, trim
from type_hints import RecDataset
from utils import set_seed
from utils_classes.generators import SequenceGenerator, SkippableGenerator


def model_sensitivity(sequences: SkippableGenerator, model: SequentialRecommender, position: int, log_path: str = "model_sensitivity.csv"):
    """
    The experiments consists in taking a source sequence `x` with a label `y`, result of `model(x)`.
    Then replace the element at position `position` of the sequence with each element of the alphabet (given by the `dataset`), 
    generating a new sequence x', and see the percentage of elements such that: 
    model(x') != y.

    The percentage reflects the sensitivity of the sequential recommender model on the position`position`of the sequence.

    Args:
        sequences: A generator that yields the sequences.
        model: the sequential recommender model we want to test the sensitivity on 
        dataset: the dataset used to train the sequential recommender, which will be used to take the alphabet.
    """
    seen_idx = set()
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        seen_idx = set(log["position"].tolist())

    print(f"[DEBUG] seen_idx: {seen_idx}")

    if position in seen_idx:
        print(f"[DEBUG] skipping position {position}")
        return

    if ConfigParams.DATASET == RecDataset.ML_1M:
        alphabet = torch.tensor(list(range(1, NumItems.ML_1M.value)))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")
    i = 0
    start_i, end_i = 0, 400
    avg = set()
    for i, sequence in enumerate(tqdm(sequences, desc=f"Testing model sensitivity on position {position}", total=end_i-start_i)):
        if i < start_i:
            continue
        if i >= end_i:
            break
        x = trim(sequence.squeeze(0)).unsqueeze(0)

        equal, changed = 0, 0
        if x.size(1) <= position:
            continue
        gt = model(x).argmax(-1).item()

        x_primes = x.repeat(len(alphabet), 1)
        assert x_primes.shape == torch.Size([len(alphabet), x.size(1)]), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
        positions = torch.tensor([position] * len(alphabet))
        
        x_primes[torch.arange(len(alphabet)), positions] = alphabet
        x_primes = pad_batch(x_primes, MAX_LENGTH)

        ys = model(x_primes).argmax(-1)
        result = ys == gt
        equal = result.sum().item()
        changed = (len(result) - equal)
        # print(f"[i: {i}] Position: {position}, Equal: {equal}, changed: {changed}")
        avg.add(equal / (equal + changed))
    print(f"Avg sequences which have the same label after changing an item at position {position} are: {(mean(avg) * 100):3f}%")
    log_run(position=position, avg=mean(avg), num_seqs=end_i-start_i, save_path=log_path)

def log_run(position: int, avg: float, num_seqs: int, save_path: str):
    df = pd.DataFrame({})
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            df = pd.read_csv(f)

    data = {"position": [position],
            "num_seqs": [num_seqs],
            "avg": [avg * 100]}

    new_df = pd.DataFrame(data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(save_path, index=False)

def main(config_path: Optional[str]=None, log_path: str="results/model_sensitivity.csv"):
    if config_path:
        ConfigParams.reload(config_path)
        ConfigParams.fix()
    print(ConfigParams.configs_dict())
    set_seed()
    config = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    sequences = SequenceGenerator(config)
    model = generate_model(config)
    # both ends included
    start_i, end_i = 49, 0
    for i in tqdm(range(start_i, end_i-1, -1), "Testing model sensitivity on all positions"):
        model_sensitivity(model=model, sequences=sequences, position=i, log_path=log_path)

if __name__ == "__main__":
    fire.Fire(main)






