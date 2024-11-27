"""
Evalutes the automata learned from the dataset of good and bad points.

The idea is to generate another dataset of good and bad points, limiting the
usable characters to the automa's alphabet, and compute the evaluation metrics
by computing true/false positive/negatives on the good and bad points.
"""

import json
import warnings
from typing import Optional

import fire
import pandas as pd
import toml
from aalpy.automata.Dfa import Dfa
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import os
from torch import Tensor
from tqdm import tqdm

from automata_learning.learning import learning_pipeline
from automata_learning.utils import run_automata
from config import ConfigParams
from genetic.dataset.generate import generate
from genetic.dataset.utils import dataset_difference
from models.config_utils import generate_model, get_config
from models.utils import trim
from performance_evaluation.alignment.utils import (log_run, pk_exists,
                                                    preprocess_interaction)
from performance_evaluation.evaluation_utils import (compute_metrics,
                                                     print_confusion_matrix)
from type_hints import GoodBadDataset
from utils import set_seed
from utils_classes.generators import DatasetGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def generate_test_dataset(source_sequence: Tensor, model:SequentialRecommender, dfa: Dfa) -> GoodBadDataset:
    """
    Generate a `GoodBadDataset` using only the items in the dfa alphabet for the mutations.
    This will ensure that the sequences in the dataset are always recognized by the automata.

    Args:
        source_sequence: The source sequence for the dataset generation.
        model: The black box model for the dataset generation
        dfa: The dfa on which the alphabet will be taken.

    Returns:
        A GoodBadDataset tuple.
    """
    alphabet = [c for c in dfa.get_input_alphabet() if isinstance(c, int)]
    return generate(source_sequence, model, alphabet)


def evaluate(dfa: Dfa, test_dataset: GoodBadDataset):
    """
    Evaluates the DFA over the test_dataset, returning the tp, fp, tn and fn
    metrics.
    """
    good, bad = test_dataset  
    tp, fp, tn, fn = 0,0,0,0
    for (sequence, _) in good:
        x = trim(sequence.squeeze(0)).tolist()
        accepts = run_automata(dfa, x)
        if accepts:
            tp += 1
        else:
            fn += 1
    for (sequence, _) in bad:
        x = trim(sequence.squeeze(0)).tolist()
        rejects = not run_automata(dfa, x)
        if rejects:
            tn += 1
        else:
            fp += 1
    return tp, fp, tn, fn

def evaluate_all(datasets: DatasetGenerator, 
                 oracle: SequentialRecommender,
                 end_i: int,
                 log_path: Optional[str]=None):

    prev_df = pd.DataFrame({})

    primary_key = ["source_sequence"]
    if log_path and os.path.exists(log_path):
        prev_df = pd.read_csv(log_path)

    assert not prev_df.duplicated().any()
    
    pbar = tqdm(desc="Automata Learning performance evaluation...", leave=False, total=end_i)
    i=0
    while i < end_i:
        pbar.update(1)
        next_sequence = preprocess_interaction(datasets.interactions.peek())
        next_sequence_str = ",".join([str(c) for c in next_sequence])
        config_dict = ConfigParams.configs_dict()
        new_row = pd.DataFrame({"source_sequence": [next_sequence_str], **config_dict})
        temp_df = pd.concat([prev_df, new_row], ignore_index=True)
        if pk_exists(df=temp_df, primary_key=primary_key.copy(), consider_config=True):
            #TODO: this doesn't  work
            print(f"[{i}] Skipping source sequence {next_sequence} since it still exists in the log with the same config")
            i += 1
            datasets.skip()
            continue
        else:
            dataset, interaction = next(datasets)
            source_sequence = preprocess_interaction(interaction)
            pbar.set_postfix_str(f"On sequence: {",".join([str(c) for c in next_sequence])}")
            assert isinstance(source_sequence, list) and (all(isinstance(x, Tensor) for x in source_sequence) or all(isinstance(x, int) for x in source_sequence))
            dfa = learning_pipeline(source_sequence, dataset)
            test_dataset = generate_test_dataset(interaction, oracle, dfa)

            # Remove from test the examples that come from test
            test_dataset = (dataset_difference(test_dataset[0], dataset[0]),
                            dataset_difference(test_dataset[1], dataset[1]))

            tp, fp, tn, fn = evaluate(dfa, test_dataset)
            precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
            print("----------------------------------------")
            print(f"[{i}] Precision: {precision}")
            print(f"[{i}] Accuracy: {accuracy}")
            print(f"[{i}] Recall: {recall}")
            print_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)
            print("----------------------------------------")
            train_dataset_len=len(dataset[0]), len(dataset[1])
            test_dataset_len=len(test_dataset[0]), len(test_dataset[1])
            
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
                    "source_sequence": ",".join([str(c) for c in source_sequence])}
            if log_path:
                prev_df = log_run(log=log, prev_df=prev_df, save_path=log_path, primary_key=["source_sequence"])
            i+=1

def main(use_cache: bool = False, 
         config_path: Optional[str]=None, 
         end_i: int=30, 
         log_path: Optional[str]=None):
    set_seed()
    ConfigParams.reload(config_path)
    ConfigParams.fix()
    config = get_config(dataset=ConfigParams().DATASET, model=ConfigParams().MODEL)
    oracle: SequentialRecommender = generate_model(config)
    datasets = DatasetGenerator(config=config, use_cache=use_cache, return_interaction=True)

    params = {
            "parameters":
            {
                "use_cache": use_cache,
                "end_i": end_i,
                }}
    print(f"""
          -----------------------
          CONFIG
          -----------------------
          ---Inputs---
          {json.dumps(params, indent=2)}
          ---Config.toml---
          {json.dumps(toml.load(ConfigParams._config_path), indent=2)}
          -----------------------
          """)

    evaluate_all(datasets=datasets, 
                 oracle=oracle, 
                 end_i=end_i,
                 log_path=log_path)


if __name__ == "__main__":
    fire.Fire(main)
