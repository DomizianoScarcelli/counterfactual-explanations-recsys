from deap.base import deepcopy
from tqdm import tqdm

from config import ConfigParams
from constants import PADDING_CHAR
from genetic.dataset.generate import generate
from genetic.dataset.utils import (are_dataset_equal, dataset_difference,
                                   get_dataloaders, interaction_to_tensor)
from genetic.utils import NumItems
from models.config_utils import generate_model, get_config
from type_hints import RecDataset
from utils_classes.generators import SequenceGenerator


def test_RangeOfItemsIsCorrect():
    config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    train_data, eval_data, test_data = get_dataloaders(config)
    items = set()

    correct_set = {}
    if ConfigParams.DATASET == RecDataset.ML_1M:
        correct_set = set(range(1, NumItems.ML_1M.value)) | {PADDING_CHAR}
    else:
        raise NotImplementedError(f"Test not implemented for dataset {ConfigParams.DATASET}")

    for interaction in tqdm(train_data):
        if isinstance(interaction, tuple):
            interaction = interaction[0]
        seq = interaction_to_tensor(interaction).squeeze()
        items |= set(seq.flatten().tolist())
        if items == correct_set:
            return
    
    for interaction in tqdm(test_data):
        if isinstance(interaction, tuple):
            interaction = interaction[0]
        seq = interaction_to_tensor(interaction).squeeze()
        items |= set(seq.flatten().tolist())
        if items == correct_set:
            return

    for interaction in tqdm(eval_data):
        if isinstance(interaction, tuple):
            interaction = interaction[0]
        seq = interaction_to_tensor(interaction).squeeze()
        items |= set(seq.flatten().tolist())
        if items == correct_set:
            return

    with open("data/universe.txt", "w") as f:
        f.write(str(items))

    assert items == correct_set, f"Range of items is not correct. Lengths are: {len(items)} and {len(correct_set)}, difference: {(correct_set - items) if len(correct_set) > len(items) else items - correct_set}"

class TestGenerators:
    def test_sequence_generator(self):
        config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
        sequences = SequenceGenerator(config)
        for seq in sequences:
            if 0 in seq:
                assert seq.squeeze().tolist().count(0) == 1, f"Sequence unpadded incorrectly"


def test_dataset_determinism():
    """
    Tests if the dataset generation algorithm is deterministic, meaning the
    same source sequence should always generate the same dataset
    """
    config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    sequences = SequenceGenerator(config)
    model = generate_model(config)
    i = 0
    while True:
        try:
            sequence = next(sequences)
        except StopIteration:
            break
        if i > 10:
            break
        dataset = generate(deepcopy(sequence), deepcopy(model))
        other_dataset = generate(deepcopy(sequence), deepcopy(model))
        good, bad = dataset
        o_good, o_bad = other_dataset
        
        assert are_dataset_equal(good, o_good), f"good != o_good. Difference length is {max(len(dataset_difference(good, o_good)), len(dataset_difference(o_good, good)))}"
        assert are_dataset_equal(bad, o_bad)


class TestLimitedAlphabet:
    # TODO: test that when dataset generation with a limited alphabet is
    # performed, no other symbols between the symbols in the source sequence
    # and the ones in the limited alphabet are in the dataset
    pass
