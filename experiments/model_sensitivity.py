from typing import Generator

from genetic.utils import NumItems
from constants import MAX_LENGTH

import os
import pandas as pd
import fire
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm
from statistics import mean
from models.utils import trim, pad, pad_batch

from config import DATASET, MODEL
from genetic.dataset.generate import sequence_generator
from models.config_utils import generate_model, get_config
from type_hints import RecDataset
from utils import set_seed


def model_sensitivity(sequences: Generator, model: SequentialRecommender, position: int):
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
    set_seed()
    if DATASET == RecDataset.ML_1M:
        alphabet = list(range(NumItems.ML_1M.value))
    else:
        raise NotImplementedError(f"Dataset {DATASET} not supported yet!")
    i = 0
    start_i, end_i = 0, 50
    pbar = tqdm(total=end_i - start_i, desc=f"Testing model sensitivity on position {position}", leave=False)
    avg = set()
    while True:
        #TODO: this trick can maybe be avoided since the sequence generator isn't so expensive as the dataset generator
        if i < start_i:
            pbar.update(1)
            i += 1
            continue
        if i > end_i:
            break
        try:
            x = trim(next(sequences).squeeze(0)).unsqueeze(0)
        except StopIteration:
            break
        
        equal, changed = 0, 0
        if x.size(1) <= position:
            i += 1
            continue
        gt = model(x).argmax(-1).item()
        #TODO: create a big tensor with all the x_primes and then input it in batch to the model
        x_primes = x.repeat(len(alphabet), 1)
        assert x_primes.shape == torch.Size([len(alphabet), x.size(1)]), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
        alphabet = torch.tensor(alphabet)
        positions = torch.tensor([position] * len(alphabet))
        
        x_primes[torch.arange(len(alphabet)), positions] = alphabet
        x_primes = pad_batch(x_primes, MAX_LENGTH)
        
        ys = model(x_primes).argmax(-1)
        result = ys == gt
        equal = result.sum().item()
        changed = (len(result) - equal)
        # print(f"[i: {i}] Position: {position}, Equal: {equal}, changed: {changed}")
        avg.add(equal / (equal + changed))
        i+=1
        pbar.update(1)
    print(f"Avg for position {position} is: {mean(avg) * 100}, run logged!")
    log_run(position=position, avg=mean(avg))

def log_run(position: int, avg: float, save_path: str="model_sensitivity.csv"):
    df = pd.DataFrame({})
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            df = pd.read_csv(f)

    data = {"position": [position],
            "avg": [avg * 100]}
    new_df = pd.DataFrame(data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(save_path, index=False)

def main():
    config = get_config(dataset=DATASET, model=MODEL)
    sequences = sequence_generator(config)
    model = generate_model(config)
    # both ends included
    start_i, end_i = 47, 0
    for i in tqdm(range(start_i, end_i-1, -1), "Testing model sensitivity on all positions"):
        model_sensitivity(model=model, sequences=sequences, position=i)

if __name__ == "__main__":
    fire.Fire(main)






