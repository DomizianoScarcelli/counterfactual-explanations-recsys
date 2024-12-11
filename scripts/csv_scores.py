from collections import Counter
from typing import List, Tuple

import fire
import pandas as pd
import torch
from torch import Tensor

from config import ConfigParams
from generation.utils import get_items
from sensitivity.utils import category_scores, print_topk_info
from type_hints import RecDataset
from utils import seq_tostr


def scores_from_csv(csv_path: str, print_info: bool = False):
    df = pd.read_csv(csv_path)
    # just keep the first occurrence of the sequence
    df = df.drop_duplicates(subset=["i"], keep="first")

    mapping = {}
    for _, row in df.iterrows():
        header = "source"
        seq = [int(char) for char in row[header].split(",")]  # type: ignore
        cat_count, dscores = category_scores(seq)
        mapping[seq_tostr(seq)] = {"cat_count": cat_count, "dscores": dscores}

        if print_info:
            print_topk_info(seq, cat_count, dscores)

    print(f"Mapping is: {mapping}")
    return mapping


def counterfactual_scores(seq: List[int], position: int, alphabet: Tensor):
    # Generate matrix of all possible changes over the position and alphabet
    og_scores = category_scores(seq)
    x = torch.tensor(seq).unsqueeze(0)
    x_primes = x.repeat(len(alphabet), 1)
    assert x_primes.shape == torch.Size(
        [len(alphabet), x.size(1)]
    ), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
    positions = torch.tensor([position] * len(alphabet))
    x_primes[torch.arange(len(alphabet)), positions] = alphabet

    cat_count, dscores = Counter(), Counter()
    for cseq in x_primes:
        result = category_scores(cseq.tolist())
        cat_count += result[0]
        dscores += result[1]

    cat_count = Counter(
        {key: value / len(alphabet) for key, value in cat_count.items()}
    )  # round and normalize
    dscores = Counter(
        {key: value / len(alphabet) for key, value in dscores.items()}
    )  # round and normalize

    og_cat_count, og_dscores = og_scores

    cat_count_delta = og_cat_count - cat_count
    dscores_delta = og_dscores - dscores

    # Aggregate score into a single counter
    # Print info
    print(f"-" * 50)
    # print("Og cat count", og_cat_count)
    # print("Og dscores", og_dscores)
    # print(f"+" * 50)
    # print("Counter cat count", cat_count)
    # print("Counter dscores", dscores)
    # print(f"+" * 50)
    print("Delta cat count", cat_count_delta)
    print("Delta dscores", dscores_delta)
    print(f"-" * 50)
    return cat_count, dscores


def counterfactual_scores_from_csv(csv_path: str, print_info: bool = False):
    df = pd.read_csv(csv_path)
    # just keep the first occurrence of the sequence
    df = df.drop_duplicates(subset=["i"], keep="first")
    if ConfigParams.DATASET == RecDataset.ML_1M:
        alphabet = torch.tensor(list(get_items()))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")

    mapping = {}
    for i, (_, row) in enumerate(df.iterrows()):
        if i > 10:
            break
        header = "source"
        seq = [int(char) for char in row[header].split(",")]  # type: ignore
        cat_count, dscores = counterfactual_scores(
            seq, position=len(seq) - 1, alphabet=alphabet
        )
        mapping[seq_tostr(seq)] = {"cat_count": cat_count, "dscores": dscores}

        # if print_info:
        #     print_topk_info(seq, cat_count, dscores)

    # print(f"Mapping is: {mapping}")


if __name__ == "__main__":
    # python -m scripts.csv_scores --csv_path="results/evaluate/alignment/different_splits_run_cats.csv"
    # fire.Fire(scores_from_csv)
    fire.Fire(counterfactual_scores_from_csv)
