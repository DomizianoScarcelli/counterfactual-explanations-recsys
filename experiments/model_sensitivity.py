import os
from statistics import mean
from typing import List, Literal, Optional, Set

import fire
import pandas as pd
import torch
from pandas import DataFrame
from recbole.model.abstract_recommender import SequentialRecommender
from torch import Tensor
from tqdm import tqdm

from config import ConfigParams
from genetic.utils import get_category_map, get_items
from models.config_utils import generate_model, get_config
from models.utils import topk, trim
from performance_evaluation.alignment.utils import get_log_stats, log_run, stats_to_df
from type_hints import RecDataset
from utils import set_seed
from utils_classes.distances import jaccard_sim, ndcg_at, precision_at
from utils_classes.generators import SequenceGenerator, SkippableGenerator


def generate_matrix_of_possible_sequences(
    sequence: Tensor, model: SequentialRecommender, position: int, alphabet: Tensor
):

    x = trim(sequence.squeeze(0)).unsqueeze(0)

    if x.size(1) <= position:
        return

    out = model(x)

    x_primes = x.repeat(len(alphabet), 1)
    assert x_primes.shape == torch.Size(
        [len(alphabet), x.size(1)]
    ), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
    positions = torch.tensor([position] * len(alphabet))

    x_primes[torch.arange(len(alphabet)), positions] = alphabet
    out_primes = model(x_primes)

    return out, out_primes


def model_sensitivity_category(
    sequences: SkippableGenerator,
    model: SequentialRecommender,
    position: int,
    log_path: Optional[str] = None,
):

    def cat_all_changes(gt: Set[str], new: Set[str]):
        """
        Returns True if all the labels in gt are changed in new, False otherwise
            gt: {Horror, Thriller}
            new: {Drama}
            changes: True

            gt: {Horror, Thriller}
            new: {Drama, Thriller}
            changes: False
        """
        return not gt & new

    def cat_at_least_changes(gt: Set[str], new: Set[str]):
        """
        Returns True if at least one of the the labels in gt are changed in new, False otherwise
            gt: [Horror, Triller]
            new: [Drama]
            changes: True

            gt: [Horror, Triller]
            new: [Drama, Triller]
            changes: True
        """
        return len(gt - new) != 0

    category_map = get_category_map()

    seen_idx = set()
    prev_df = pd.DataFrame({})
    if log_path and os.path.exists(log_path):
        prev_df = pd.read_csv(log_path)
        seen_idx = set(prev_df["position"].tolist())

    print(f"[DEBUG] seen_idx: {seen_idx}")

    if position in seen_idx:
        print(f"[DEBUG] skipping position {position}")
        return

    if ConfigParams.DATASET == RecDataset.ML_1M:
        alphabet = torch.tensor(list(get_items()))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")
    i = 0
    start_i, end_i = 0, 500
    pbar = tqdm(
        total=end_i - start_i, desc=f"Testing model sensitivity on position {position}"
    )
    i_list, all_changes, any_changes, jaccards, sequence_list = [], [], [], [], []
    for i, sequence in enumerate(sequences):
        if i < start_i:
            continue
        if i >= end_i:
            break

        # TODO: don't know if this is working
        # if pk_exists(prev_df, primary_key=["i", "position"], consider_config=True):
        #     # print(f"Skipping i: {i} at pos: {position}...")
        #     continue

        pbar.update(1)

        result = generate_matrix_of_possible_sequences(
            sequence, model, position, alphabet
        )
        if not result:
            continue
        out, out_primes = result

        out_cat: Set[str] = set(category_map[out.argmax(-1).item()])

        out_primes_cat: List[Set[str]] = [
            set(category_map[out_prime.argmax(-1).item()]) for out_prime in out_primes
        ]

        all_change = mean(
            cat_all_changes(gt=out_cat, new=out_prime_k)
            for out_prime_k in out_primes_cat
        )
        any_change = mean(
            cat_at_least_changes(gt=out_cat, new=out_prime_k)
            for out_prime_k in out_primes_cat
        )

        jaccard = mean(
            jaccard_sim(a=out_cat, b=out_prime_k) for out_prime_k in out_primes_cat
        )
        i_list.append(i)
        all_changes.append(all_change)
        any_changes.append(any_change)
        jaccards.append(jaccard)
        sequence_list.append(sequence.squeeze())

    if log_path:
        data = {
            "i": i_list,
            "position": [position] * len(i_list),
            "all_changes": [v * 100 for v in all_changes],
            "any_changes": [v * 100 for v in any_changes],
            "jaccards": [v * 100 for v in jaccards],
            "sequence": [",".join(str(v) for v in x) for x in sequence_list],
            "model": [ConfigParams.MODEL.value] * len(i_list),
            "dataset": [ConfigParams.DATASET.value] * len(i_list),
        }

        prev_df = log_run(
            prev_df=prev_df,
            log=data,
            save_path=log_path,
            add_config=True,
            primary_key=["i", "position"],
        )

        # prev_df = prev_df.sort_values(by="i")


# TODO: reflect changes from the categorized one
def model_sensitivity_simple(
    sequences: SkippableGenerator,
    model: SequentialRecommender,
    position: int,
    k: int = 1,
    log_path: Optional[str] = None,
):
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
    if log_path and os.path.exists(log_path):
        prev_df = pd.read_csv(log_path)
        filtered_df = prev_df[prev_df["k"] == k]
        seen_idx = set(filtered_df["position"].tolist())

    print(f"[DEBUG] seen_idx: {seen_idx}")

    if position in seen_idx:
        print(f"[DEBUG] skipping position {position}")
        return

    if ConfigParams.DATASET == RecDataset.ML_1M:
        alphabet = torch.tensor(list(get_items()))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")
    i = 0
    start_i, end_i = 0, 100
    pbar = tqdm(
        total=end_i - start_i, desc=f"Testing model sensitivity on position {position}"
    )
    for i, sequence in enumerate(sequences):
        if i < start_i:
            continue
        if i >= end_i:
            break
        pbar.update(1)

        result = generate_matrix_of_possible_sequences(
            sequence, model, position, alphabet
        )
        if not result:
            continue
        out, out_primes = result

        # TODO: since I'm using metrics@k, I don't think I need this
        out_k = topk(out, k, dim=-1, indices=True).squeeze()  # [K]
        out_primes_k = topk(out_primes, k, dim=-1, indices=True)  # [len(alphabet), K]

        if k == 1:
            out_k = out_k.unsqueeze(-1)

        jaccards = mean(
            jaccard_sim(a=out_k, b=(out_prime_k.squeeze() if k != 1 else out_prime_k))
            for out_prime_k in out_primes_k
        )
        precisions = mean(
            precision_at(
                k=k, a=out_k, b=(out_prime_k.squeeze() if k != 1 else out_prime_k)
            )
            for out_prime_k in out_primes_k
        )
        ndcgs = mean(
            ndcg_at(k=k, a=out_k, b=out_prime_k.squeeze() if k != 1 else out_prime_k)
            for out_prime_k in out_primes_k
        )
        # pbar.set_postfix_str(
        #     f"jacc: {mean(jaccards)*100:.2f}, prec: {mean(precisions)*100:.2f}, ndcg: {mean(ndcgs)*100:.2f}"
        # )

        if log_path:
            data = {
                "i": i,
                "sequence": ",".join([str(x) for x in sequence]),
                "position": position,
                "num_seqs": end_i - start_i,
                "mean_precision": precisions * 100,
                "mean_ndcgs": ndcgs * 100,
                "mean_jaccard": jaccards * 100,
                "k": k,
                "model": ConfigParams.MODEL.value,
                "dataset": ConfigParams.DATASET.value,
            }

            prev_df = log_run(
                prev_df=prev_df, log=data, save_path=log_path, add_config=True
            )


def get_all_stats(log_path: str, metrics: List[str]) -> DataFrame:
    """
    Given a dataframe result of a model sensitivity experiment, it returns a dataframe of the type
    ```csv
    position,num_seqs,mean_precision,mean_ndcgs,mean_jaccard,k,model,dataset,determinism,pop_size,generations,halloffame_ratio,allowed_mutations,timestamp
    49,100,31.062,34.446,21.233,10,BERT4Rec,ml-1m,True,2048,10,0.2,"('replace', 'swap', 'add', 'delete', 'shuffle', 'reverse')","Tue, 03 Dec 2024 10:38:26"
    48,100,47.649,52.966,34.73,10,BERT4Rec,ml-1m,True,2048,10,0.2,"('replace', 'swap', 'add', 'delete', 'shuffle', 'reverse')","Tue, 03 Dec 2024 10:38:26"
    47,100,58.732,64.677,45.015,10,BERT4Rec,ml-1m,True,2048,10,0.2,"('replace', 'swap', 'add', 'delete', 'shuffle', 'reverse')","Tue, 03 Dec 2024 10:38:26"
    ...
    ```
    For each sequence, where the the mean is grouped by the same position over different sequences.
    """
    stats = get_log_stats(log_path=log_path, group_by=["position"], metrics=metrics)
    df = stats_to_df(stats)
    return df


def get_sequence_stats(log_path: str, metrics: List[str]) -> DataFrame:
    """
    Given a dataframe result of a model sensitivity experiment, it returns a dataframe of the type

    ```csv
    position,sequence,num_seqs,mean_precision,mean_ndcgs,mean_jaccard,k,model,dataset,determinism,pop_size,generations,halloffame_ratio,allowed_mutations,timestamp
    49,"1,2,3,4,5,6",100,31.062,34.446,21.233,10,BERT4Rec,ml-1m,True,2048,10,0.2,"('replace', 'swap', 'add', 'delete', 'shuffle', 'reverse')","Tue, 03 Dec 2024 10:38:26"
    49,"1,2,3,4,5,7",100,31.062,34.446,21.233,10,BERT4Rec,ml-1m,True,2048,10,0.2,"('replace', 'swap', 'add', 'delete', 'shuffle', 'reverse')","Tue, 03 Dec 2024 10:38:26"
    48,"1,2,3,4,5,6",100,47.649,52.966,34.73,10,BERT4Rec,ml-1m,True,2048,10,0.2,"('replace', 'swap', 'add', 'delete', 'shuffle', 'reverse')","Tue, 03 Dec 2024 10:38:26"
    48,"1,2,3,4,5,7",100,47.649,52.966,34.73,10,BERT4Rec,ml-1m,True,2048,10,0.2,"('replace', 'swap', 'add', 'delete', 'shuffle', 'reverse')","Tue, 03 Dec 2024 10:38:26"
    ...
    For each sequence, where the the mean is grouped by the same sequence over different positions.
    """
    stats = get_log_stats(log_path=log_path, group_by=["sequence"], metrics=metrics)
    df = stats_to_df(stats)
    return df


def main(
    config_path: Optional[str] = None,
    log_path: Optional[str] = None,
    k: int = 1,
    target: Literal["item", "category"] = "item",
    mode: Literal["evaluate", "stats"] = "evaluate",
    groupby: Optional[Literal["sequence", "position"]] = None,
    stats_save_path: Optional[str] = None,
):
    if config_path:
        ConfigParams.reload(config_path)
        ConfigParams.fix()
    if mode == "evaluate":
        print(ConfigParams.configs_dict())
        set_seed()
        config = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
        sequences = SequenceGenerator(config)
        model = generate_model(config)
        # both ends included
        start_i, end_i = 49, 0
        for i in tqdm(
            range(start_i, end_i - 1, -1), "Testing model sensitivity on all positions"
        ):
            if target == "item":
                model_sensitivity_simple(
                    model=model, sequences=sequences, position=i, log_path=log_path, k=k
                )
            elif target == "category":
                model_sensitivity_category(
                    model=model,
                    sequences=sequences,
                    position=i,
                    log_path=log_path,
                )
            else:
                raise ValueError(f"target must be 'item' or 'category', not '{target}'")
    elif mode == "stats":
        if target == "item":
            metrics = []  # TODO: to be defined
        elif target == "category":
            metrics = ["all_changes", "any_changes", "jaccards"]
        else:
            raise ValueError(f"target must be 'item' or 'category', not '{target}'")

        if not log_path:
            raise ValueError(f"define a log_path as a source for the stats")
        if groupby == "sequence":
            stats = get_sequence_stats(log_path, metrics)
        elif groupby == "position":
            stats = get_all_stats(log_path, metrics)
        else:
            raise ValueError(
                f"groupby must be 'sequence' or 'position', not '{groupby}'"
            )
        if stats is not None:
            print(stats)

        if stats is not None and stats_save_path:
            stats.to_csv(stats_save_path, index=False)
    else:
        raise ValueError(f"mode must be 'evaluate' or 'stats', not '{mode}'")


# if __name__ == "__main__":
#     fire.Fire(main)
