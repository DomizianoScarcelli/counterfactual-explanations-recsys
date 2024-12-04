import os
from statistics import mean
from typing import Optional

import fire
import pandas as pd
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from config import ConfigParams
from genetic.utils import Items, get_items
from models.config_utils import generate_model, get_config
from models.utils import topk, trim
from performance_evaluation.alignment.utils import log_run
from type_hints import RecDataset
from utils import set_seed
from utils_classes.distances import jaccard_sim, ndcg_at, precision_at
from utils_classes.generators import SequenceGenerator, SkippableGenerator


def model_sensitivity(sequences: SkippableGenerator, 
                      model: SequentialRecommender, 
                      position: int,
                      k: int = 1,
                      log_path: str = "model_sensitivity.csv"):
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
    prev_df = pd.DataFrame({})
    if os.path.exists(log_path):
        prev_df = pd.read_csv(log_path)
        filtered_df = prev_df[prev_df['k'] == k]
        seen_idx = set(filtered_df["position"].tolist())

    print(f"[DEBUG] seen_idx: {seen_idx}")

    if position in seen_idx:
        print(f"[DEBUG] skipping position {position}")
        return

    if ConfigParams.DATASET == RecDataset.ML_1M:
        alphabet = torch.tensor(list(get_items(Items.ML_1M)))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")
    i = 0
    start_i, end_i = 0, 100
    precisions, ndcgs, jaccards = set(), set(), set()
    pbar = tqdm(total=end_i-start_i, desc=f"Testing model sensitivity on position {position}")
    for i, sequence in enumerate(sequences):
        if i < start_i:
            continue
        if i >= end_i:
            break
        pbar.update(1)
        x = trim(sequence.squeeze(0)).unsqueeze(0)

        if x.size(1) <= position:
            continue
        out = model(x)
        out_k = topk(out, k, dim=-1, indices=True).squeeze() #[K]
        if k == 1:
            out_k = out_k.unsqueeze(-1)

        x_primes = x.repeat(len(alphabet), 1)
        assert x_primes.shape == torch.Size([len(alphabet), x.size(1)]), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
        positions = torch.tensor([position] * len(alphabet))
        
        x_primes[torch.arange(len(alphabet)), positions] = alphabet

        out_primes = model(x_primes)
        out_primes_k = topk(out_primes, k, dim=-1, indices=True) #[len(alphabet), K]

        jaccards.add(mean(jaccard_sim(a=out_k, b=(out_prime_k.squeeze() if k != 1 else out_prime_k)) for out_prime_k in out_primes_k))
        precisions.add(mean(precision_at(k=k, a=out_k, b=(out_prime_k.squeeze() if k != 1 else out_prime_k)) for out_prime_k in out_primes_k))
        ndcgs.add(mean(ndcg_at(k=k, a=out_k, b=out_prime_k.squeeze()) for out_prime_k in out_primes_k))
        pbar.set_postfix_str(f"jacc: {mean(jaccards)*100:.2f}, prec: {mean(precisions)*100:.2f}, ndcg: {mean(ndcgs)*100:.2f}")

    data = {"position": position,
            "num_seqs": end_i-start_i,
            "mean_precision": mean(precisions) * 100,
            "mean_ndgs": mean(ndcgs)* 100, 
            "mean_jaccard": mean(jaccards) * 100, 
            "k": k,
            "model": ConfigParams.MODEL.value,
            "dataset": ConfigParams.DATASET.value}

    prev_df = log_run(prev_df=prev_df, log=data,  save_path=log_path, add_config=True)


def main(config_path: Optional[str]=None, log_path: str="results/model_sensitivity.csv", k: int=1):
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
        model_sensitivity(model=model, sequences=sequences, position=i, log_path=log_path, k=k)

if __name__ == "__main__":
    fire.Fire(main)






