from deap import base, creator, tools, algorithms
import numpy as np
import random
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import torch.nn.functional as F
from enum import Enum
import _pickle as cPickle
from deap_generator import GeneticGenerationStrategy
import pytest
from model_funcs import model_predict, model_batch_predict

@pytest.fixture()
def sequence():
    return

@pytest.fixture()
def model():
    return
@pytest.fixture()
def interaction():
    return

def test_generate(sequence, model, interaction, model_predict):
    good_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                     predictor=lambda x: model_predict(seq=x,
                                                                   interaction=interaction,
                                                                   model=model,
                                                                   prob=True),
                                                     pop_size=100,
                                                     good_examples=True,
                                                     generations=10)
