import pytest
from automata_learning import (run_automata)
from trace_alignment import (create_intersection_automata, 
                             trace_alignment, 
                             align)

#----------TESTS WITH MOCK DATA--------------#
@pytest.mark.skip()
def test_augmented_trace_automata_mock(mock_t_dfa, mock_t_dfa_aug, mock_original_trace, mock_edited_trace):
    # mock_t_dfa.visualize("saved_automatas/mock_t_dfa")
    # mock_t_dfa_aug.visualize("saved_automatas/mock_t_dfa_aug")

    t_dfa_accepts = run_automata(mock_t_dfa, mock_original_trace)
    assert t_dfa_accepts, "T_DFA rejected good point"

    t_dfa_aug_accepts = run_automata(mock_t_dfa_aug, mock_edited_trace)
    assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

@pytest.mark.skip()
def test_augmented_constraint_automata_mock(mock_a_dfa, mock_a_dfa_aug, mock_original_trace, mock_edited_trace): 
    # mock_a_dfa.visualize("saved_automatas/mock_a_dfa")
    # mock_a_dfa_aug.visualize("saved_automatas/mock_a_dfa_aug")

    a_dfa_accepts = run_automata(mock_a_dfa, mock_original_trace)
    assert a_dfa_accepts, "A_DFA rejected good point"

    a_dfa_accepts = run_automata(mock_a_dfa_aug, mock_edited_trace)
    assert a_dfa_accepts, "A_DFA rejected edited good point"

@pytest.mark.skip()
def test_run_trace_alignment_bad_trace_mock(mock_a_dfa_aug, mock_bad_trace):
    a_dfa_aug_accepts = run_automata(mock_a_dfa_aug, mock_bad_trace)
    assert not a_dfa_aug_accepts, f"Bad trace should be accepted"
    
    # TODO: if a character cannot be read by the automata, everything
    # collapses, see how to handle this
    alignment, cost = trace_alignment(mock_a_dfa_aug, mock_bad_trace)
    print(f"Best alignment {alignment} with cost {cost}")

@pytest.mark.skip()
def test_trace_alignment_single_mock(mock_a_dfa_aug, mock_bad_trace):
    aligned_trace, _ = trace_alignment(mock_a_dfa_aug, mock_bad_trace)
    aligned_accepts = run_automata(mock_a_dfa_aug, aligned_trace)
    assert aligned_accepts, "Automa should accept aligned trace"
    original_rejects = not run_automata(mock_a_dfa_aug, mock_bad_trace)
    assert original_rejects, "Automa should reject original bad trace"

# @pytest.mark.skip()
def test_trace_alignment_mock(mock_a_dfa_aug, mock_dataset):
    _, bp = mock_dataset
    for bad_trace, _ in bp:
        test_trace_alignment_single_mock(mock_a_dfa_aug, bad_trace)
    

#----------TESTS WITH REAL DATA--------------#

@pytest.mark.skip()
def test_augmented_trace_automata(t_dfa, t_dfa_aug, original_trace, edited_trace):
    t_dfa_accepts = run_automata(t_dfa, original_trace)
    assert t_dfa_accepts, "T_DFA rejected good point"

    t_dfa_aug_accepts = run_automata(t_dfa_aug, edited_trace)
    assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

@pytest.mark.skip()
def test_augmented_constraint_automata(a_dfa, a_dfa_aug, original_trace, edited_trace): 
    a_dfa_accepts = run_automata(a_dfa, original_trace)
    assert a_dfa_accepts, "A_DFA rejected good point"

    a_dfa_accepts = run_automata(a_dfa_aug, edited_trace)
    assert a_dfa_accepts, "A_DFA rejected edited good point"

@pytest.mark.skip(reason="Planning automata is not being used right now")
def test_create_planning_automata(a_dfa_aug, t_dfa_aug, original_trace, edited_trace):
    planning_dfa = create_intersection_automata(a_dfa_aug, t_dfa_aug)

    a_dfa_aug_accepts = run_automata(a_dfa_aug, original_trace)
    t_dfa_aug_accepts = run_automata(t_dfa_aug, original_trace)
    planning_dfa_accepts = run_automata(planning_dfa, original_trace)

    assert a_dfa_aug_accepts and t_dfa_aug_accepts and planning_dfa_accepts, f"""
    DFA are not accepting good input
        a_dfa_aug_accepts: {a_dfa_aug_accepts}
        t_dfa_aug_accepts: {t_dfa_aug_accepts}
        planning_dfa_accepts: {planning_dfa_accepts}
    """
    t_dfa_aug_accepts = run_automata(t_dfa_aug, edited_trace)
    a_dfa_aug_accepts = run_automata(a_dfa_aug, edited_trace)
    planning_dfa_accepts = run_automata(planning_dfa, edited_trace)

    assert a_dfa_aug_accepts and t_dfa_aug_accepts and planning_dfa_accepts, f"""
    DFA are not accepting edited good input
        a_dfa_aug_accepts: {a_dfa_aug_accepts}
        t_dfa_aug_accepts: {t_dfa_aug_accepts}
        planning_dfa_accepts: {planning_dfa_accepts}
    """
    print("Planning DFA alphabet:", planning_dfa.get_input_alphabet())


# @pytest.mark.skip()
def test_trace_alignment_single(a_dfa_aug, bad_trace):
    aligned_trace, _ = trace_alignment(a_dfa_aug, bad_trace)
    aligned_accepts = run_automata(a_dfa_aug, aligned_trace)
    assert aligned_accepts, "Automa should accept aligned trace"
    original_rejects = not run_automata(a_dfa_aug, bad_trace)
    assert original_rejects, "Automa should reject original bad trace"

@pytest.mark.skip()
def test_trace_alignment(a_dfa_aug, dataset):
    _, bp = dataset
    for bad_trace, _ in bp:
        test_trace_alignment_single(a_dfa_aug, bad_trace)


#----------GENERAL TESTS--------------#
def test_align():
    trace = [1,2,3,5,6]
    alignment = ("sync_1", "del_2", "add_4", "sync_3", "sync_5", "del_6")
    aligned_trace = align(trace, alignment)
    
    correct_alignment = [1,4,3,5]
    assert aligned_trace == correct_alignment, f"""
    Aligned trace is wrong
    corect: {correct_alignment}
    computed: {aligned_trace}
    """


    

    
