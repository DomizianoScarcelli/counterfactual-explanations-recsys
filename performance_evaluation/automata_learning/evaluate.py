"""
Evalutes the automata learned from the dataset of good and bad points.

The idea is to generate another dataset of good and bad points, limiting the
usable characters to the automa's alphabet, and compute the evaluation metrics
by computing true/false positive/negatives on the good and bad points.
"""

from tqdm import tqdm
from aalpy.automata.Dfa import Dfa
from recbole.model.abstract_recommender import SequentialRecommender
from automata_learning.learning import learning_pipeline
from genetic.dataset.generate import generate
from torch import Tensor
import fire
from typing import Generator, Tuple
from genetic.dataset.utils import get_sequence_from_interaction
from type_hints import GoodBadDataset
from utils import set_seed
from performance_evaluation.evaluation_utils import compute_metrics, print_confusion_matrix

import fire
from recbole.model.abstract_recommender import SequentialRecommender

from config import DATASET, MODEL
from genetic.dataset.generate import dataset_generator, interaction_generator
from models.config_utils import generate_model, get_config
from utils import set_seed
from models.utils import trim
from automata_learning.utils import run_automata

def generate_test_dataset(source_sequence: Tensor, model:SequentialRecommender, dfa: Dfa) -> GoodBadDataset:
    alphabet = [c for c in dfa.get_input_alphabet() if isinstance(c, int)]
    return generate(source_sequence, model, alphabet)[0]


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

def evaluate_all(interactions: Generator, 
                 datasets: Generator, 
                 oracle: SequentialRecommender,
                 num_counterfactuals: int=10):
    tp, fp, tn, fn = 0,0,0,0
    for i, ((train_dataset, _), interaction) in enumerate(tqdm(zip(datasets, interactions), desc="Automata Learning performance evaluation...")):
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break

        source_sequence = trim(get_sequence_from_interaction(interaction).squeeze(0)).tolist()
        print(f"source_sequence is: ", source_sequence)
        dfa = learning_pipeline(source_sequence, train_dataset)
        test_dataset = generate_test_dataset(interaction, oracle, dfa)
        ntp, nfp, ntn, nfn = evaluate_single(dfa, test_dataset)
        tp += ntp
        fp += nfp
        tn += ntn
        fn += nfn
        precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        print("----------------------------------------")
        print(f"[{i}] Precision: {precision}")
        print(f"[{i}] Accuracy: {accuracy}")
        print(f"[{i}] Recall: {recall}")
        print_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)
        print("----------------------------------------")


def main(use_cache: bool = False):
    set_seed()
    config = get_config(dataset=DATASET, model=MODEL)
    oracle: SequentialRecommender = generate_model(config)
    interactions = interaction_generator(config)
    datasets = dataset_generator(config=config, use_cache=use_cache)
    evaluate_all(interactions=interactions,
                 datasets=datasets, 
                 oracle=oracle)



if __name__ == "__main__":
    fire.Fire(main)
