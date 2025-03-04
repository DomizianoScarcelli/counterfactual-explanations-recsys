"""
This files aggregate all the test_determinism functions in all the other test
files, in order to easily test the determinism of all random functions.
"""

from utils.utils import SeedSetter

SeedSetter.set_seed()

# TODO: to fully test determinism, for each function generate the needed variables, reset the seed + generate again the needed variables and asser they are equal.

from tests.automata_learning.test_automata_learning import \
    test_automata_learning_determinism
from tests.genetic.test_genetic import TestGeneticDeterminism
