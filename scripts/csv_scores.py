from utils_classes.generators import SequenceGenerator
import os
import warnings
from statistics import mean
from typing import List, Optional, Tuple
import json

import fire
import pandas as pd
import torch
from tqdm import tqdm

from config import ConfigParams
from generation.utils import get_items
from models.config_utils import get_config
from models.utils import trim
from sensitivity.utils import (
    compute_scores,
    counterfactual_scores_deltas,
    print_topk_info,
)
from type_hints import RecDataset
from utils import seq_tostr

warnings.simplefilter(action="ignore", category=FutureWarning)


def scores_from_csv(csv_path: str, print_info: bool = False):
    """Generate category and score metrics for sequences stored in a CSV file.

    This function reads a CSV file containing sequences, computes category and score metrics for each sequence,
    and stores the results in a mapping dictionary. Only the first occurrence of each sequence (based on a unique
    identifier column) is considered.

    Args:
        csv_path (str): Path to the CSV file. The file must contain a column `source` with sequences as
                        comma-separated integers and a column `i` for identifying unique sequences.
        print_info (bool, optional): If True, prints detailed information for each sequence,
                                     including top-k category and score metrics. Defaults to False.

    Returns:
        dict: A dictionary mapping each sequence (as a string) to a dictionary containing:
              - "cat_count": The category count metrics (Counter).
              - "dscores": The score metrics (Counter).

    Notes:
        - The function ignores duplicate sequences based on the `i` column, keeping only the first occurrence.
        - The `category_scores` function is used to compute the metrics for each sequence.
        - The `seq_tostr` function converts the sequence list into a string for use as a dictionary key.
    """
    df = pd.read_csv(csv_path)
    # just keep the first occurrence of the sequence
    df = df.drop_duplicates(subset=["i"], keep="first")

    mapping = {}
    for _, row in df.iterrows():
        header = "source" if "source" in row else "sequence"
        seq = [int(char) for char in row[header].split(",")]  # type: ignore
        cat_count, dscores = compute_scores(torch.tensor(seq))
        mapping[seq_tostr(seq)] = {"cat_count": cat_count, "dscores": dscores}

        if print_info:
            print_topk_info(seq, cat_count, dscores)

    # print(f"[DEBUG] Mapping is: {mapping}")
    return mapping


def scores_from_sequences(range_i: Tuple[int, int], print_info: bool = False):
    mapping = {}
    config = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    sequences = SequenceGenerator(config)
    start, end = range_i
    for _ in range(start):
        sequences.skip()
    for _ in range(start, end):
        seq = sequences.next()
        seq = trim(seq.squeeze())
        cat_count, dscores = compute_scores(seq)
        mapping[seq_tostr(seq)] = {"cat_count": cat_count, "dscores": dscores}

        if print_info:
            print_topk_info(seq.tolist(), cat_count, dscores)

    # print(f"[DEBUG] Mapping is: {mapping}")
    return mapping


def counterfactual_scores_from_csv(
    csv_path: str,
    position: int,
    join_on: Optional[
        Tuple[str, str] | str | Tuple[List[str], List[str]] | List[str]
    ] = None,
    select: Optional[List[str]] = None,
):
    """Compute counterfactual metrics for sequences in a CSV file.

    This function processes sequences stored in a CSV file, computes counterfactual category and score metrics
    by modifying the last position in each sequence with values from a predefined alphabet, and stores the results
    in a mapping dictionary. Only the first occurrence of each sequence (based on a unique identifier column)
    is considered.

    Args:
        csv_path (str): Path to the CSV file. The file must contain a column `source` with sequences as
                        comma-separated integers and a column `i` for identifying unique sequences.
        print_info (bool, optional): If True, prints detailed information for each sequence,
                                     including top-k category and score metrics. Defaults to False.

    Returns:
        dict: A dictionary mapping each sequence (as a string) to a dictionary containing:
              - "cat_count": The counterfactual category count metrics (Counter).
              - "dscores": The counterfactual score metrics (Counter).

    Notes:
        - The `ConfigParams.DATASET` variable determines the dataset being processed. Currently, only the
          `ML_1M` dataset is supported, and the alphabet is constructed using `get_items()`.
        - The function considers up to the first 10 sequences in the file for counterfactual evaluation.
        - The `counterfactual_scores` function is used to compute metrics for each sequence.
        - The `seq_tostr` function converts the sequence list into a string for use as a dictionary key.

    Raises:
        NotImplementedError: If the dataset specified in `ConfigParams.DATASET` is not supported.
    """
    og_df = pd.read_csv(csv_path)
    # just keep the first occurrence of the sequence
    df = og_df.drop_duplicates(subset=["i"], keep="first")
    if ConfigParams.DATASET in [RecDataset.ML_1M, RecDataset.ML_100K]:
        alphabet = torch.tensor(list(get_items()))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")

    # Extract sequences
    header = "source" if "source" in df.columns else "sequence"
    sequences = df[header].str.split(",").apply(lambda x: [int(i) for i in x])
    # sequences = df[header].apply(lambda x: ast.literal_eval(x))
    trimmed_sequences = sequences.apply(lambda seq: trim(torch.tensor(seq)))

    # Compute counterfactual metrics for all sequences

    def compute_metrics(seq):
        if len(seq) <= position:
            return None, None
        cat_count_delta, dscores_delta = counterfactual_scores_deltas(
            seq, position=position, alphabet=alphabet
        )
        return mean(cat_count_delta.values()), mean(dscores_delta.values())

    tqdm.pandas(total=500)  # TODO: compute the total
    metrics = trimmed_sequences.progress_apply(compute_metrics)
    df.loc[:, "cc_delta"], df.loc[:, "ds_delta"] = zip(*metrics)
    df.loc[:, "position"] = position
    df["sequence"] = trimmed_sequences.apply(seq_tostr)

    if join_on:
        if isinstance(join_on, str) or isinstance(join_on, list):
            left_on, right_on = join_on, join_on
        elif isinstance(join_on, tuple):
            if len(join_on) != 2:
                raise ValueError(
                    f"join_on should be a str or a tuple of two strings, not {join_on} of type {type(join_on)}"
                )
            left_on, right_on = join_on
        else:
            raise ValueError(
                f"join_on should be a str or a tuple of two strings, not {join_on} of type {type(join_on)}"
            )

        df = og_df.merge(
            right=df,
            left_on=left_on,
            right_on=right_on,
            how="inner",
            suffixes=("", "_right"),
        )

        # Drop the columns from the right DataFrame that have the same name as the left DataFrame's columns
        columns_to_drop = [col for col in df.columns if col.endswith("_right")]
        df = df.drop(columns=columns_to_drop)
    if select:
        df = df[select]

    return df


def main(
    csv_path: Optional[str] = None,
    on_counterfactuals: bool = False,
    join_on: Optional[Tuple[str, str] | str] = None,
    save_path: Optional[str] = None,
    select: Optional[List[str]] = None,
    print_info: bool = False,
):
    if on_counterfactuals and csv_path:
        start_i, end_i = 49, 0
        result_df = pd.DataFrame({})
        for position in tqdm(range(start_i, end_i - 1, -1)):
            df = counterfactual_scores_from_csv(
                csv_path=csv_path, join_on=join_on, position=position, select=select
            )
            result_df = pd.concat([result_df, df])

            if save_path:
                if save_path == "same":
                    base, ext = os.path.splitext(csv_path)
                    result_df.to_csv(f"{base}_scores{ext}")
                else:
                    result_df.to_csv(save_path)
            if print_info:
                print(result_df.head())
    else:
        if csv_path:
            mapping = scores_from_csv(csv_path=csv_path, print_info=print_info)
        else:
            mapping = scores_from_sequences(range_i=(0, 50), print_info=print_info)

        if save_path:
            with open(save_path, "w") as f:
                json.dump(mapping, f)
            print(f"Mappings saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
