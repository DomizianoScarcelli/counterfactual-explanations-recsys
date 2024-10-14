from deap import base, creator, tools, algorithms
import numpy as np
import random
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import torch.nn.functional as F
from enum import Enum
import _pickle as cPickle
from deap_generator import edit_distance
from deap_generator import GeneticGenerationStrategy
import pytest

@pytest.fixture()
def sequence():
    return

@pytest.fixture()
def model():
    return
@pytest.fixture()
def interaction():
    return


def test_edit_distance():
    seq1 = torch.tensor([1,2,3,4,5])
    seq2 = torch.tensor([1,3,2])
    seq3 = torch.tensor([2,3,4,5,6])
    edit_distance(seq1, seq2)
    edit_distance(seq2, seq1)
    edit_distance(seq1, seq3)

@pytest.mark.skip()
def test_generate(sequence, model, interaction, model_predict):
    good_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                     predictor=lambda x: model_predict(seq=x,
                                                                   interaction=interaction,
                                                                   model=model,
                                                                   prob=True),
                                                     pop_size=100,
                                                     good_examples=True,
                                                     generations=10)
