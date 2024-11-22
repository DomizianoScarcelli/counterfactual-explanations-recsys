from genetic.dataset.utils import are_dataset_equal
from type_hints import GoodBadDataset
from utils_classes.generators import DatasetGenerator, InteractionGenerator, SequenceGenerator
from recbole.trainer import Interaction
import torch
from torch import Tensor


def interaction_equality(i1: Interaction, i2: Interaction):
    i1, i2 = i1.interaction, i2.interaction #type: ignore
    for key1, key2 in zip(i1, i2):
        value1, value2 = i1[key1], i2[key2]
        if isinstance(value1, Tensor) and isinstance(value2, Tensor):
            if not torch.all(value1 == value2):
                return False
        elif value1 != value2:
                return False
        else:
            print("What?")
            return False
    return True

def test_interaction_generator(config):
    interactions = InteractionGenerator(config)

    num_skips = 10
    for _ in range(num_skips):
        interactions.skip()

    assert interactions.index == num_skips

    interaction_at_num = next(interactions)

    interactions.reset()
    assert interactions.index == 0
    
    interaction: Interaction
    for i, interaction in enumerate(interactions):
        if i == num_skips:
            assert interaction_equality(interaction, interaction_at_num)

def test_sequence_generator(config):    
    sequences = SequenceGenerator(config)

    num_skips = 10
    for _ in range(num_skips):
        sequences.skip()

    assert sequences.index == num_skips

    sequence_at_num = next(sequences)

    sequences.reset()
    assert sequences.index == 0
    
    sequence: Interaction
    for i, sequence in enumerate(sequences):
        if i == num_skips:
            assert torch.all(sequence == sequence_at_num)

def test_dataset_generator(config):
    datasets = DatasetGenerator(config, use_cache=False)

    num_skips = 1
    for _ in range(num_skips):
        datasets.skip()

    assert datasets.index == num_skips

    good_at_num, bad_at_num = next(datasets)

    datasets.reset()
    assert datasets.index == 0
    
    dataset: GoodBadDataset
    for i, dataset in enumerate(datasets):
        if i != num_skips:
            good, bad = dataset
            
            assert not are_dataset_equal(good, good_at_num)
            assert not are_dataset_equal(bad, bad_at_num)

        if i == num_skips:
            good, bad = dataset
            
            assert are_dataset_equal(good, good_at_num)
            assert are_dataset_equal(bad, bad_at_num)
            
            break

    

