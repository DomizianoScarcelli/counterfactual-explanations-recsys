"""
Evalutes the automata learned from the dataset of good and bad points.

The idea is to generate another dataset of good and bad points, limiting the
usable characters to the automa's alphabet, and compute the evaluation metrics
by computing true/false positive/negatives on the good and bad points.
"""

from typing import List, Tuple

import fire
import pandas as pd
from aalpy.automata.Dfa import Dfa
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import os
from torch import Tensor
from tqdm import tqdm
from performance_evaluation.alignment.utils import preprocess_interaction

from automata_learning.learning import learning_pipeline
from automata_learning.utils import run_automata
from config import DATASET, GENERATIONS, HALLOFFAME_RATIO, MODEL, POP_SIZE
from genetic.dataset.generate import generate
from genetic.dataset.utils import dataset_difference
from models.config_utils import generate_model, get_config
from models.utils import trim
from performance_evaluation.evaluation_utils import (compute_metrics,
                                                     print_confusion_matrix)
from type_hints import GoodBadDataset
from utils import set_seed
from utils_classes.generators import DatasetGenerator, SkippableGenerator
import warnings

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


def evaluate_single(dfa: Dfa, test_dataset: GoodBadDataset):
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

def evaluate_all(datasets: SkippableGenerator, 
                 oracle: SequentialRecommender,
                 num_counterfactuals: int=30):
    for i, (dataset, interaction) in enumerate(tqdm(datasets, desc="Automata Learning performance evaluation...")):
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        
        source_sequence = preprocess_interaction(interaction)
        assert isinstance(source_sequence, list) and (all(isinstance(x, Tensor) for x in source_sequence) or all(isinstance(x, int) for x in source_sequence))
        print(f"source_sequence is: ", source_sequence)
        dfa = learning_pipeline(source_sequence, dataset)
        test_dataset = generate_test_dataset(interaction, oracle, dfa)

        print(f"[DEBUG] Test dataset length: {len(test_dataset[0]) + len(test_dataset[1])}")
        # Remove from test the examples that come from test
        test_dataset = (dataset_difference(test_dataset[0], dataset[0]),
                        dataset_difference(test_dataset[1], dataset[1]))
        print(f"[DEBUG] Test dataset length: {len(test_dataset[0]) + len(test_dataset[1])}")

        tp, fp, tn, fn = evaluate_single(dfa, test_dataset)
        precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        print("----------------------------------------")
        print(f"[{i}] Precision: {precision}")
        print(f"[{i}] Accuracy: {accuracy}")
        print(f"[{i}] Recall: {recall}")
        print_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)
        print("----------------------------------------")
        log_run(metrics=(tp, fp, tn, fn),
                train_dataset_len=(len(dataset[0]),
                                   len(dataset[1])),
                test_dataset_len=(len(test_dataset[0]),
                                  len(test_dataset[1])),
                source_sequence=source_sequence)

def log_run(metrics: Tuple[int, int, int, int],
            train_dataset_len: Tuple[int, int],
            test_dataset_len: Tuple[int, int],
            source_sequence: List[int],
            save_path: str = "automata_learning_eval.csv"):
    old_log = pd.DataFrame({})
    if os.path.exists(save_path):
        old_log = pd.read_csv(save_path)

    tp, fp, tn, fn = metrics
    precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
    data = {"tp": [tp],
            "fp": [fp],
            "tn": [tn],
            "fn": [fn],
            "precision": [precision * 100],
            "accuracy": [accuracy * 100],
            "recall": [recall * 100],
            "train_dataset_len": [train_dataset_len],
            "test_dataset_len": [test_dataset_len],
            "source_sequence_len": [len(source_sequence)],
            "num_generations": [GENERATIONS],
            "population_size": [POP_SIZE],
            "halloffame_ratio": [HALLOFFAME_RATIO],
            "dataset": [DATASET.value],
            "model": [MODEL.value],
            "source_sequence": [source_sequence]}

    new_df = pd.DataFrame(data)
    df = pd.concat([old_log, new_df], ignore_index=True)
    df.to_csv(save_path, index=False)
    return df

def evaluate_stats():
    pass

def main(use_cache: bool = False):
    set_seed()
    config = get_config(dataset=DATASET, model=MODEL)
    oracle: SequentialRecommender = generate_model(config)
    datasets = DatasetGenerator(config=config, use_cache=use_cache, return_interaction=True)
    evaluate_all(datasets=datasets, 
                 oracle=oracle)



if __name__ == "__main__":
    fire.Fire(main)
