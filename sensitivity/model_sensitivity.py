import os
from statistics import mean
from typing import List, Literal, Optional, Set

import pandas as pd
import torch
from pandas import DataFrame
from recbole.model.abstract_recommender import SequentialRecommender
from torch import Tensor
from tqdm import tqdm

from config import ConfigParams
from constants import cat2id
from generation.utils import equal_ys, get_category_map, get_items
from models.config_utils import generate_model, get_config
from models.utils import topk, trim
from performance_evaluation.alignment.utils import get_log_stats, log_run, stats_to_df
from type_hints import CategorySet, RecDataset
from utils import seq_tostr, set_seed
from utils_classes.distances import (
    jaccard_sim,
    DEPRECATED_ndng_at,
    ndcg,
    pairwise_jaccard_sim,
    precision_at,
)
from utils_classes.generators import SequenceGenerator, SkippableGenerator


def generate_sequence_variants(sequence: Tensor, position: int, alphabet: Tensor):
    """
    Generate a matrix of possible sequences by modifying a single position in the input sequence.

    This function takes an input sequence, modifies a specific position using each element in the given alphabet,
    and passes both the original and modified sequences through the provided model. It returns the model's output
    for the original sequence and the modified sequences.

    Args:
        sequence (Tensor): The input sequence as a tensor of shape `(1, seq_length)`.
        model (SequentialRecommender): The model used to generate outputs for the sequences.
        position (int): The index of the position in the sequence to be modified.
        alphabet (Tensor): A tensor containing the values to be used for modification.

    Returns:
        Tuple[Tensor, Tensor]:
            - `out`: The model's output for the original sequence.
            - `out_primes`: The model's outputs for the modified sequences, with shape `(len(alphabet), output_dim)`.

    Notes:
        - The `trim` function is assumed to preprocess the sequence by trimming unwanted padding or values.
        - The function ensures that the `position` is within the bounds of the sequence.
        - The tensor `x_primes` is created by repeating the original sequence for each element in the alphabet
          and substituting the value at the specified `position` with the corresponding alphabet value.

    Example:
        Given `sequence = [[1, 2, 3]]`, `position = 1`, and `alphabet = [4, 5]`:
        - The modified sequences are `[[1, 4, 3], [1, 5, 3]]`.
        - The function returns the model's output for `[[1, 2, 3]]` and `[[1, 4, 3], [1, 5, 3]]`.
    """
    x = sequence
    assert x.dim() == 2 and x.size(0) == 1

    if x.size(1) <= position:
        return

    x_primes = x.repeat(len(alphabet), 1)
    assert x_primes.shape == torch.Size(
        [len(alphabet), x.size(1)]
    ), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
    positions = torch.tensor([position] * len(alphabet))

    x_primes[torch.arange(len(alphabet)), positions] = alphabet

    return x_primes


def model_sensitivity_category(
    sequences: SkippableGenerator,
    model: SequentialRecommender,
    position: int,
    k: int,
    log_path: Optional[str] = None,
):
    """
    Analyze the sensitivity of a sequential recommender model to changes in category predictions
    when input sequences are modified at a specific position. Optionally logs the results to a file.

    Parameters:
        sequences (SkippableGenerator): A generator that provides sequences to analyze.
        model (SequentialRecommender): The recommender model to evaluate.
        position (int): The index in the sequence where modifications will be made.
        k (int): The number of top predictions to consider for sensitivity analysis.
        log_path (Optional[str]): Path to a CSV file for logging results. If provided, results
                                  are appended to the file; otherwise, they are printed.

    Returns:
        None

    Description:
        - The function evaluates the model's sensitivity to changes in the categories of its top-k predictions
          when the sequence is modified at the specified `position`.
        - Each sequence is modified by replacing the character at `position` with all possible characters
          in the `alphabet`.
        - The function computes the following metrics:
          - **All Changes**: Proportion of categories in the original predictions that are completely replaced.
          - **Any Changes**: Proportion of categories in the original predictions that are partially replaced.
          - **Jaccard Similarity**: Similarity measure between the original and modified category sets.

    Output:
        - Logs or prints metrics for each sequence, including the proportion of changes and similarity scores.
        - If `log_path` is specified, results are saved with additional context, including the dataset and model used.

    Notes:
        - Skips sequences that have already been processed if a `log_path` is provided with existing results.
        - Supports only the MovieLens 1M dataset (`ML_1M`). Throws a `NotImplementedError` for other datasets.

    """

    def cat_all_changes(gt: Set, new: Set):
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

    def cat_at_least_changes(gt: Set, new: Set):
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

    itemid2cat = get_category_map()

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
    start_i, end_i = 0, 200
    pbar = tqdm(
        total=end_i - start_i, desc=f"Testing model sensitivity on position {position}"
    )
    count = 0
    i_list, jaccards, sequence_list, counterfactuals, ndcgs = ([], [], [], [], [])
    for i, sequence in enumerate(sequences):
        if i < start_i:
            continue
        if i >= end_i:
            break

        count += 1

        # TODO: fix pk_exists before removing the comment
        # if pk_exists(prev_df, primary_key=["i", "position"], consider_config=True):
        #     # print(f"Skipping i: {i} at pos: {position}...")
        #     continue

        pbar.update(1)

        sequence = trim(sequence.squeeze(0)).unsqueeze(0)
        x_primes = generate_sequence_variants(sequence, position, alphabet)
        if x_primes is None:
            continue

        out = model(sequence)
        out_primes = model(x_primes)

        topk_out = topk(logits=out, k=k, dim=-1, indices=True).squeeze(0)  # [k]
        out_cat: List[CategorySet] = [
            set(cat2id[cat] for cat in itemid2cat[y.item()]) for y in topk_out  # type: ignore
        ]  # [k]

        topk_out_primes = topk(
            logits=out_primes, k=k, dim=-1, indices=True
        )  # [n_items, k]

        # TODO: this and out_cat must be rivisited, is the content really what you need?
        out_primes_cat: List[List[CategorySet]] = [
            [set(cat2id[cat] for cat in itemid2cat[y.item()]) for y in topk_out_prime]  # type: ignore
            for topk_out_prime in topk_out_primes
        ]  # [n_items, k]

        # TODO: This can be vectorized by using torch operations
        jaccard, counterfactual, ndcg_v = [], [], []
        n_items = len(out_primes_cat)
        for n_i in range(n_items):
            y = out_cat
            y_prime = out_primes_cat[n_i]

            assert len(y) == len(y_prime) == k

            jaccard.append(pairwise_jaccard_sim(y, y_prime))
            counterfactual.append(not equal_ys(y, y_prime))
            ndcg_v.append(ndcg(y, y_prime))

        i_list.append(i)
        jaccards.append(mean(jaccard))
        ndcgs.append(mean(ndcg_v))
        counterfactuals.append(mean(counterfactual))
        sequence_list.append(sequence.squeeze().tolist())
        pbar.set_postfix_str(
            f"jacc: {mean(jaccards)*100:.2f}%, ndcg: {mean(ndcgs) * 100:.2f}, counterfactuals: {mean(counterfactuals)*100:.2f}%"
        )

    data = {
        "i": i_list,
        "position": [position] * len(i_list),
        "count": [count] * len(i_list),
        "k": [k] * len(i_list),
        "jaccards": [v * 100 for v in jaccards],  # similarity
        "ndcg": [v * 100 for v in ndcgs],  # the higher, the more similar
        "counterfactuals": counterfactuals,
        "alphabet_len": [len(alphabet)] * len(i_list),
        "sequence": [seq_tostr(x) for x in sequence_list],
        "model": [ConfigParams.MODEL.value] * len(i_list),
        "dataset": [ConfigParams.DATASET.value] * len(i_list),
    }
    if log_path:
        prev_df = log_run(
            prev_df=prev_df,
            log=data,
            save_path=log_path,
            add_config=True,
            primary_key=["i", "position", "k"],
        )
    else:
        print(pd.DataFrame(data))


# TODO: reflect changes from the categorized one, this must be heavily changed.
def model_sensitivity_simple(
    sequences: SkippableGenerator,
    model: SequentialRecommender,
    position: int,
    k: int = 1,
    log_path: Optional[str] = None,
):
    """
    The sensitivity consists in taking a source sequence `x` with a label `y`, result of `model(x)`.
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
    start_i, end_i = 0, 20
    pbar = tqdm(
        total=end_i - start_i, desc=f"Testing model sensitivity on position {position}"
    )
    for i, sequence in enumerate(sequences):
        if i < start_i:
            continue
        if i >= end_i:
            break
        pbar.update(1)

        result = generate_sequence_variants(sequence, model, position, alphabet)
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
            DEPRECATED_ndng_at(
                k=k, a=out_k, b=out_prime_k.squeeze() if k != 1 else out_prime_k
            )
            for out_prime_k in out_primes_k
        )
        # pbar.set_postfix_str(
        #     f"jacc: {mean(jaccards)*100:.2f}, prec: {mean(precisions)*100:.2f}, ndcg: {mean(ndcgs)*100:.2f}"
        # )

        if log_path:
            data = {
                "i": i,
                "sequence": seq_tostr(sequence),
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


def get_stats(
    log_path: str, metrics: List[str], groupby: List[str], orderby: Optional[List[str]]
) -> DataFrame:
    stats = get_log_stats(log_path=log_path, group_by=groupby, metrics=metrics)
    df = stats_to_df(stats)
    if orderby:
        for col in orderby:
            df[col] = pd.to_numeric(df[col], errors="ignore", downcast="integer")
        df = df.sort_values(by=orderby)
    return df


def main(
    config_path: Optional[str] = None,
    log_path: Optional[str] = None,
    k: int = 1,
    target: Literal["item", "category"] = "item",
    mode: Literal["evaluate", "stats"] = "evaluate",
    groupby: Optional[List[str]] = None,
    orderby: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    stats_save_path: Optional[str] = None,
):
    # TODO: merge this with the `cli.stats` method, this is cli logic, shouldn't be here.
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
                    k=k,
                    log_path=log_path,
                )
            else:
                raise ValueError(f"target must be 'item' or 'category', not '{target}'")
    elif mode == "stats":
        if not target and not metrics:
            raise ValueError("target or metrics should be set to something")

        if target == "item" and not metrics:
            metrics = []  # TODO: to be defined

        if target == "category" and not metrics:
            metrics = ["all_changes", "any_changes", "jaccards"]

        assert metrics

        if not log_path:
            raise ValueError(f"define a log_path as a source for the stats")

        if not groupby:
            raise ValueError(f"group_by should not be None: {groupby}")
        stats = get_stats(
            log_path=log_path, groupby=groupby, metrics=metrics, orderby=orderby
        )
        if stats is not None:
            print(stats)

        if stats is not None and stats_save_path:
            stats.to_csv(stats_save_path, index=False)
    else:
        raise ValueError(f"mode must be 'evaluate' or 'stats', not '{mode}'")
