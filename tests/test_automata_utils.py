from automata_learning.utils import invert_automata, run_automata


def test_invert_automata(mock_a_dfa_aug, mock_original_trace):
    assert run_automata(mock_a_dfa_aug, mock_original_trace), "Original automata should accept good trace"
    invert_automata(mock_a_dfa_aug)
    assert not run_automata(mock_a_dfa_aug, mock_original_trace), "Inverted automata should not accept good trace"
