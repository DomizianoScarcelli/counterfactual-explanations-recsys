"""
This files aggregate all the test_determinism functions in all the other test
files, in order to easily test the determinism of all random functions.
"""

# from tests.genetic.test_genetic import TestGeneticDeterminism
# from tests.automata_learning.test_automata_learning import test_automata_learning_determinism
# from tests.models.test_ExtendedBERT4Rec import test_model_determinism
from tests.dataset.test_generate import test_dataset_determinism


