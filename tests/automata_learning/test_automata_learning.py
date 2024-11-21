from automata_learning.learning import learning_pipeline
from automata_learning.utils import run_automata
from config import DATASET, MODEL
from genetic.dataset.generate import generate, sequence_generator
from models.config_utils import generate_model, get_config
from models.utils import trim
from utils import set_seed


def test_automata_accepts_source_sequence():
    """
    Test if automata accepts the source sequence
    """
    set_seed()
    config = get_config(model=MODEL, dataset=DATASET)
    model = generate_model(config)
    sequences = sequence_generator(config)
    i = 0
    while True:
        # While instead of for loop in order to be able to skip some indices
        # without generating the dataset
        if i < 12:
            i += 1
            continue
        try:
            source_trace = next(sequences)
        except StopIteration:
            break
        train_dataset, _ = generate(source_trace, model)
        trace = trim(source_trace.squeeze(0).tolist())
        assert -1 not in trace
        dfa = learning_pipeline(trace, train_dataset)
        assert run_automata(dfa, trace), f"Automata do not accept sequence {i}, {trace}"
        i += 1

def test_automata_learning_determinism():
    """
    Tests if the automata learning algorithm is deterministic, meaning the
    same source sequence with the same learning dataset should always
    generate the same DFA.
    """
    set_seed()
    config = get_config(model=MODEL, dataset=DATASET)
    sequences = sequence_generator(config)
    model = generate_model(config)
    i = 0
    while True:
        try:
            sequence = next(sequences)
        except StopIteration:
            break
        if i > 20:
            break
        train_dataset, _ = generate(sequence, model)
        sequence = sequence.squeeze(0).tolist()
        dfa = learning_pipeline(sequence, train_dataset)
        other_dfa = learning_pipeline(sequence, train_dataset)
        assert dfa == other_dfa, f"Learning is non deterministic for sequence {sequence}"

