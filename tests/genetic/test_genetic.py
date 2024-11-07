import pytest
import torch

from alignment.alignment import augment_constraint_automata
from automata_learning.learning import (generate_single_accepting_sequence_dfa,
                                        learning_pipeline)
from automata_learning.utils import run_automata
from genetic.dataset.generate import generate


@pytest.mark.heavy
def test_accepting(model, sequences):
    for i, seq in enumerate(sequences):
        if i > 20:
            break
        train, _ = generate(seq, model)
        a_dfa = learning_pipeline(seq.squeeze().tolist(), train)
        t_dfa = generate_single_accepting_sequence_dfa(seq.squeeze().tolist())
        a_dfa_aug = augment_constraint_automata(a_dfa, t_dfa)
        assert run_automata(a_dfa, seq.squeeze().tolist()), f"Automata does not accept {seq.squeeze().tolist()} at index {i}"
        assert run_automata(a_dfa_aug, seq.squeeze().tolist()), f"Augmented automata does not accept {seq.squeeze().tolist()} at index {i}"

@pytest.mark.heavy
def test_does_not_contain_source_sequence(model, sequences):
    """
    Tests if dataset contains only a single reference to the source sequence.
    """
    for i, seq in enumerate(sequences):
        if i > 20:
            break
        (good, bad), _ = generate(seq, model)
        count = 0
        for gen_seq, _ in good:
            if torch.all(gen_seq == seq):
                count += 1
            assert count < 1
        for gen_seq, _ in bad:
            if torch.all(gen_seq == seq):
                count += 1
            assert count < 1

