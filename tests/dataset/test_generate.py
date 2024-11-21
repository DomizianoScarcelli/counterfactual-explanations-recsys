from deap.base import deepcopy

from automata_learning.learning import learning_pipeline
from config import DATASET, MODEL
from genetic.dataset.generate import (dataset_generator, generate,
                                      interaction_generator,
                                      sequence_generator)
from genetic.dataset.utils import (are_dataset_equal, dataset_difference,
                                   get_sequence_from_interaction)
from genetic.utils import NumItems
from models.config_utils import generate_model, get_config
from utils import set_seed


class TestGenerators:
    def test_sequence_generator(self):
        config = get_config(model=MODEL, dataset=DATASET)
        sequences = sequence_generator(config)
        for seq in sequences:
            if 0 in seq:
                assert seq.squeeze().tolist().count(0) == 1, f"Sequence unpadded incorrectly"

    def test_interaction_generator(self):
        config = get_config(model=MODEL, dataset=DATASET)
        interactions = interaction_generator(config)
        items = set()
        for interaction in interactions:
            seq = get_sequence_from_interaction(interaction).squeeze(0).tolist()
            for i in seq:
                items.add(i)
        assert min(items) == -1 and max(items) == NumItems.ML_1M.value, f"Max should be in (-1, {NumItems.ML_1M.value}), but is: ({min(items)}, {max(items)})"

def test_dataset_determinism():
    """
    Tests if the dataset generation algorithm is deterministic, meaning the
    same source sequence should always generate the same dataset
    """
    config = get_config(model=MODEL, dataset=DATASET)
    sequences = sequence_generator(config)
    model = generate_model(config)
    i = 0
    while True:
        try:
            sequence = next(sequences)
        except StopIteration:
            break
        if i > 10:
            break
        train_dataset, _ = generate(deepcopy(sequence), deepcopy(model))
        other_train_dataset, _ = generate(deepcopy(sequence), deepcopy(model))
        good, bad = train_dataset
        o_good, o_bad = other_train_dataset
        
        assert are_dataset_equal(good, o_good), f"good != o_good. Difference length is {max(len(dataset_difference(good, o_good)), len(dataset_difference(o_good, good)))}"
        assert are_dataset_equal(bad, o_bad)


class TestLimitedAlphabet:
    # TODO: test that when dataset generation with a limited alphabet is
    # performed, no other symbols between the symbols in the source sequence
    # and the ones in the limited alphabet are in the dataset
    pass
