"""
Evalutes the automata learned from the dataset of good and bad points.

The idea is to generate another dataset of good and bad points, limiting the
usable characters to the automa's alphabet, and compute the evaluation metrics
by computing true/false positive/negatives on the good and bad points.
"""


from aalpy.automata.Dfa import Dfa
from recbole.model.abstract_recommender import SequentialRecommender
from genetic.dataset.generate import generate
from torch import Tensor


def generate_test_dataset(source_sequence: Tensor, model:SequentialRecommender, dfa: Dfa):
    pass

def evaluate_automata_learnig():
    pass
