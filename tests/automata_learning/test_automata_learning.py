from automata_learning.passive_learning import learning_pipeline
from automata_learning.utils import run_automata
from config import ConfigParams
from generation.dataset.generate import generate
from generation.dataset.utils import interaction_to_tensor
from models.config_utils import generate_model, get_config
from models.utils import trim
from utils import printd
from utils_classes.generators import DatasetGenerator, SequenceGenerator


def test_automata_accepts_source_sequence():
    """
    Test if automata accepts the source sequence
    """
    config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    model = generate_model(config)
    sequences = SequenceGenerator(config)
    i = 0
    while True:
        # While instead of for loop in order to be able to skip some indices
        # without generating the dataset
        if i < 12:
            # if i < 0:
            i += 1
            sequences.skip()
            continue
        try:
            source_trace = next(sequences)
        except StopIteration:
            break
        dataset = generate(source_trace, model)
        trace = trim(source_trace.squeeze(0).tolist())
        assert -1 not in trace
        dfa = learning_pipeline(trace, dataset)
        assert run_automata(dfa, trace), f"Automata do not accept sequence {i}, {trace}"
        i += 1
        printd(f"{i} [PASSED], automata accepts the source trace", level=1)


def test_automata_learning_determinism():
    """
    Tests if the automata learning algorithm is deterministic, meaning the
    same source sequence with the same learning dataset should always
    generate the same DFA.
    """
    config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    datasets = DatasetGenerator(config, return_interaction=True)
    i = 0
    while True:
        try:
            dataset, interaction = next(datasets)
        except StopIteration:
            break
        if i > 20:
            break
        sequence = interaction_to_tensor(interaction)
        sequence = trim(sequence.squeeze()).tolist()
        dfa = learning_pipeline(sequence, dataset)
        other_dfa = learning_pipeline(sequence, dataset)
        assert (
            dfa == other_dfa
        ), f"Learning is non deterministic for sequence {sequence}"
