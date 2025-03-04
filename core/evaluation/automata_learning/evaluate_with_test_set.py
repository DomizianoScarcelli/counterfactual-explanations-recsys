"""
Evalutes the automata learned from the dataset of good and bad points.

The idea is to generate another dataset of good and bad points, limiting the
usable characters to the automa's alphabet, and compute the evaluation metrics
by computing true/false positive/negatives on the good and bad points.
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import fire
from aalpy.automata.Dfa import Dfa
from pandas import DataFrame
from torch import Tensor
from tqdm import tqdm

from core.alignment.actions import Action, decode_action
from core.automata_learning.passive_learning import learning_pipeline
from core.automata_learning.utils import run_automata
from config.config import ConfigParams
from exceptions import EmptyDatasetError
from core.generation.dataset.generate import generate
from core.generation.dataset.utils import dataset_difference
from core.models.config_utils import get_config
from core.models.utils import trim
from core.evaluation.alignment.utils import preprocess_interaction
from core.evaluation.evaluation_utils import (compute_metrics,
                                              print_confusion_matrix)
from type_hints import GoodBadDataset
from utils.utils import printd, seq_tostr
from utils.utils import DatasetGenerator
from utils.utils import RunLogger

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def evaluate_automata_learning(i, dfa):
    return {"i": i, "automata_states": len(dfa.states)}


def generate_test_dataset(
    source_sequence: Tensor, source_generator: DatasetGenerator, dfa: Dfa
) -> GoodBadDataset:
    """
    Generate a `GoodBadDataset` using only the items in the dfa alphabet for the mutations.
    This will ensure that the sequences in the dataset are always recognized by the automata.

    Args:
        source_sequence: The source sequence for the dataset generation.
        source_generator: TODO
        dfa: The dfa on which the alphabet will be taken.

    Returns:
        A GoodBadDataset tuple.
    """
    alphabet = [
        c for c in dfa.get_input_alphabet() if decode_action(c)[0] == Action.SYNC
    ]
    return generate(
        interaction=source_sequence,
        good_strat=source_generator.good_strat,
        bad_strat=source_generator.bad_strat,
        alphabet=alphabet,
    )


def evaluate_single(dfa: Dfa, test_dataset: GoodBadDataset):
    """
    Evaluates the DFA over the test_dataset, returning the tp, fp, tn and fn
    metrics.
    """
    good, bad = test_dataset
    tp, fp, tn, fn = 0, 0, 0, 0
    for sequence, _ in good:
        x = trim(sequence.squeeze(0)).tolist()
        accepts = run_automata(dfa, x)
        if accepts:
            tp += 1
        else:
            fn += 1
    for sequence, _ in bad:
        x = trim(sequence.squeeze(0)).tolist()
        rejects = not run_automata(dfa, x)
        if rejects:
            tn += 1
        else:
            fp += 1
    return tp, fp, tn, fn


def evaluate(
    datasets: DatasetGenerator,
    end_i: int,
    save_path: Optional[Path] = None,
):

    primary_key = ["source_sequence"]
    logger = None
    if save_path:
        logger = RunLogger(
            db_path=save_path, schema=None, add_config=True, merge_cols=True
        )

    pbar = tqdm(
        desc="Automata Learning performance evaluation...", leave=False, total=end_i
    )
    for i in range(end_i):
        pbar.update(1)
        next_sequence = preprocess_interaction(datasets.interactions.peek())
        new_row = {"source_sequence": seq_tostr(next_sequence)}
        if logger and logger.exists(
            new_row,
            primary_key,
            consider_config=True,
            type_sensitive=False,
            whitelist=["target_cat"],
        ):
            printd(
                f"[{i}] Skipping source sequence {next_sequence} since it still exists in the log with the same config"
            )
            datasets.skip()
            continue
        else:
            try:
                dataset, interaction = next(datasets)
            except EmptyDatasetError as e:
                print(f"Raised error {type(e)}")
                datasets.skip()
                datasets.match_indices()  # type: ignore
                log = {
                    "tp": None,
                    "fp": None,
                    "tn": None,
                    "fn": None,
                    "precision": None,
                    "accuracy": None,
                    "recall": None,
                    "train_dataset_len": None,
                    "test_dataset_len": None,
                    "source_sequence_len": None,
                    "source_sequence": seq_tostr(next_sequence),
                    "error": "EmptyDatasetError",
                }
                if logger:
                    logger.log_run(log, primary_key, tostr=True)
                continue
            source_sequence = preprocess_interaction(interaction)
            pbar.set_postfix_str(f"On sequence: {seq_tostr(next_sequence)}")
            assert isinstance(source_sequence, list) and (
                all(isinstance(x, Tensor) for x in source_sequence)
                or all(isinstance(x, int) for x in source_sequence)
            )
            dfa = learning_pipeline(source_sequence, dataset)
            try:
                test_dataset = generate_test_dataset(interaction, datasets, dfa)
            except EmptyDatasetError as e:
                print(f"Raised error {type(e)}")
                datasets.skip()
                datasets.match_indices()  # type: ignore
                continue

            # Remove from test the examples that come from test
            test_dataset = (
                dataset_difference(test_dataset[0], dataset[0]),
                dataset_difference(test_dataset[1], dataset[1]),
            )

            tp, fp, tn, fn = evaluate_single(dfa, test_dataset)
            precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
            print("----------------------------------------")
            print(f"[{i}] Precision: {precision}")
            print(f"[{i}] Accuracy: {accuracy}")
            print(f"[{i}] Recall: {recall}")
            print_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)
            print("----------------------------------------")
            train_dataset_len = len(dataset[0]), len(dataset[1])
            test_dataset_len = len(test_dataset[0]), len(test_dataset[1])

            log = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision * 100, 2),
                "accuracy": round(accuracy * 100, 2),
                "recall": round(recall * 100, 2),
                "train_dataset_len": train_dataset_len,
                "test_dataset_len": test_dataset_len,
                "source_sequence_len": len(source_sequence),
                "source_sequence": seq_tostr(source_sequence),
                "error": None,
            }
            if logger:
                logger.log_run(log, primary_key, tostr=True)


def compute_automata_metrics(df: DataFrame, average_per_user: bool = False) -> dict:
    """
    Given a pandas DataFrame with at least "tp, fp, tn, fn" columns, it computes the overall
    precision, accuracy, and recall. Supports two modes:

    1. Default (False): Aggregates tp, fp, tn, and fn across all users and then computes metrics.
    2. average_per_user=True: Computes precision, accuracy, and recall per user and averages them.

    Args:
        df (DataFrame): DataFrame containing "tp", "fp", "tn", and "fn" columns.
        average_per_user (bool): If True, compute metrics per user and then average them.

    Returns:
        dict: A dictionary containing the computed precision, accuracy, and recall.
    """
    result_df = {}

    if average_per_user:
        return df[["precision", "accuracy", "recall"]].mean().to_dict()
    else:
        # Sum tp, fp, tn, fn across all users and compute metrics globally
        metrics_cols = ["tp", "fp", "tn", "fn"]
        for metrics_col in metrics_cols:
            result_df[metrics_col] = df[metrics_col].sum()

        tp, fp, tn, fn = (
            result_df["tp"],
            result_df["fp"],
            result_df["tn"],
            result_df["fn"],
        )
        precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)

        result_df["precision"] = precision
        result_df["accuracy"] = accuracy
        result_df["recall"] = recall

    return result_df


def run_automata_learning_eval(
    use_cache: bool = False,
    end_i: int = 30,
    save_path: Optional[Path] = None,
):
    params = {
        "parameters": {
            "use_cache": use_cache,
            "end_i": end_i,
        }
    }
    print(
        f"""
          -----------------------
          CONFIG
          -----------------------
          ---Inputs---
          {json.dumps(params, indent=2)}
          ---Config.toml---
          {ConfigParams.print_config(indent=2)}
          -----------------------
          """
    )
    config = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    datasets = DatasetGenerator(
        config=config,
        use_cache=use_cache,
        return_interaction=True,
        target=ConfigParams.TARGET_CAT,
    )

    evaluate(datasets=datasets, end_i=end_i, save_path=save_path)


if __name__ == "__main__":
    fire.Fire(run_automata_learning_eval)
