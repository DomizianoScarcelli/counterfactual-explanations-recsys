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
import pandas as pd
from aalpy.automata.Dfa import Dfa
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import os
from torch import Tensor
from tqdm import tqdm

from alignment.actions import Action, decode_action
from automata_learning.learning import learning_pipeline
from automata_learning.utils import run_automata
from config import ConfigDict, ConfigParams
from generation.dataset.generate import generate
from generation.dataset.utils import dataset_difference
from models.config_utils import generate_model, get_config
from models.utils import trim
from performance_evaluation.alignment.utils import (log_run, pk_exists,
                                                    preprocess_interaction)
from performance_evaluation.evaluation_utils import (compute_metrics,
                                                     print_confusion_matrix)
from type_hints import GoodBadDataset
from utils import SeedSetter, seq_tostr
from utils_classes.generators import DatasetGenerator

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


def evaluate(dfa: Dfa, test_dataset: GoodBadDataset):
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


def evaluate_all(
    datasets: DatasetGenerator,
    oracle: SequentialRecommender,
    end_i: int,
    log_path: Optional[Path] = None,
):

    prev_df = pd.DataFrame({})

    primary_key = ["source_sequence"]
    if log_path and log_path.exists():
        prev_df = pd.read_csv(log_path)

    assert not prev_df.duplicated().any()

    pbar = tqdm(
        desc="Automata Learning performance evaluation...", leave=False, total=end_i
    )
    i = 0
    while i < end_i:
        pbar.update(1)
        next_sequence = preprocess_interaction(datasets.interactions.peek())
        next_sequence_str = seq_tostr(next_sequence)
        config_dict = ConfigParams.configs_dict()
        new_row = pd.DataFrame({"source_sequence": [next_sequence_str], **config_dict})
        temp_df = pd.concat([prev_df, new_row], ignore_index=True)
        if pk_exists(df=temp_df, primary_key=primary_key.copy(), consider_config=True):
            # TODO: this doesn't  work
            print(
                f"[{i}] Skipping source sequence {next_sequence} since it still exists in the log with the same config"
            )
            i += 1
            datasets.skip()
            continue
        else:
            dataset, interaction = next(datasets)
            # print(f"[DEBUG] dataset is: {dataset}")
            source_sequence = preprocess_interaction(interaction)
            pbar.set_postfix_str(f"On sequence: {seq_tostr(next_sequence)}")
            assert isinstance(source_sequence, list) and (
                all(isinstance(x, Tensor) for x in source_sequence)
                or all(isinstance(x, int) for x in source_sequence)
            )
            dfa = learning_pipeline(source_sequence, dataset)
            test_dataset = generate_test_dataset(interaction, datasets, dfa)

            # Remove from test the examples that come from test
            test_dataset = (
                dataset_difference(test_dataset[0], dataset[0]),
                dataset_difference(test_dataset[1], dataset[1]),
            )

            tp, fp, tn, fn = evaluate(dfa, test_dataset)
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
            }
            if log_path:
                prev_df = log_run(
                    log=log,
                    prev_df=prev_df,
                    save_path=log_path,
                    primary_key=["source_sequence"],
                )
            i += 1


def get_log_stats():
    pass


def main(
    use_cache: bool = False,
    config_path: Optional[str] = None,
    config_dict: Optional[ConfigDict] = None,
    end_i: int = 30,
    save_path: Optional[Path] = None,
):
    SeedSetter.set_seed()
    if config_path and config_dict:
        raise ValueError(
            "Only one between config_path and config_dict must be set, not both"
        )
    if config_path:
        ConfigParams.reload(config_path)
    if config_dict:
        ConfigParams.override_params(config_dict)
    ConfigParams.fix()

    if save_path and save_path.exists():
        prev_df = pd.read_csv(save_path)
        future_df = pd.DataFrame(ConfigParams.configs_dict())
        df = pd.concat([prev_df, future_df], ignore_index=True)
        # TODO: for now this works only when we are generating a result for the first sequence
        # I need to extend it to more sequences, but then this has to be put inside `evaluate_all`,
        # which will be slower since the config, oracle and dataset generator are created anyways.
        if end_i == 1 and pk_exists(df, primary_key=[], consider_config=True):
            print(f"Config skipped since it already exists")
            return

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
    config = get_config(dataset=ConfigParams().DATASET, model=ConfigParams().MODEL)
    oracle: SequentialRecommender = generate_model(config)
    datasets = DatasetGenerator(
        config=config,
        use_cache=use_cache,
        return_interaction=True,
    )

    evaluate_all(datasets=datasets, oracle=oracle, end_i=end_i, log_path=save_path)


if __name__ == "__main__":
    fire.Fire(main)
