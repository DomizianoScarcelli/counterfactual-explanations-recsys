import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Tuple

import _pickle as cPickle
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from deap import algorithms, base, creator, tools

from deap_generator import GeneticGenerationStrategy, edit_distance

@pytest.mark.skip()
def test_edit_distance():
    seq1 = torch.tensor([1,2,3,4,5])
    seq2 = torch.tensor([1,3,2])
    seq3 = torch.tensor([2,3,4,5,6])
    edit_distance(seq1, seq2)
    edit_distance(seq2, seq1)
    edit_distance(seq1, seq3)

