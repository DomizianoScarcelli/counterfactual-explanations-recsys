from config import DATASET, MODEL
from genetic.dataset.generate import interaction_generator, sequence_generator
from genetic.dataset.utils import get_sequence_from_interaction
from genetic.utils import NumItems
from models.config_utils import get_config


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


