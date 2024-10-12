from copy import deepcopy
import pytest
from automata_learning import (run_automata)
from automata_utils import invert_automata
from graph_search import encode_action_str
from trace_alignment import (create_intersection_automata, 
                             trace_alignment, 
                             align, trace_disalignment)

#----------TESTS WITH MOCK DATA--------------#
# @pytest.mark.skip()
def test_augmented_trace_automata_mock(mock_t_dfa, mock_t_dfa_aug, mock_original_trace, mock_edited_trace):
    # mock_t_dfa.visualize("saved_automatas/mock_t_dfa")
    # mock_t_dfa_aug.visualize("saved_automatas/mock_t_dfa_aug")

    t_dfa_accepts = run_automata(mock_t_dfa, mock_original_trace)
    assert t_dfa_accepts, "T_DFA rejected good point"

    t_dfa_aug_accepts = run_automata(mock_t_dfa_aug, mock_edited_trace)
    assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

# @pytest.mark.skip()
def test_augmented_constraint_automata_mock(mock_a_dfa, mock_a_dfa_aug, mock_original_trace, mock_edited_trace): 
    # mock_a_dfa.visualize("saved_automatas/mock_a_dfa")
    # mock_a_dfa_aug.visualize("saved_automatas/mock_a_dfa_aug")

    a_dfa_accepts = run_automata(mock_a_dfa, mock_original_trace)
    assert a_dfa_accepts, "A_DFA rejected good point"

    a_dfa_accepts = run_automata(mock_a_dfa_aug, mock_edited_trace)
    assert a_dfa_accepts, "A_DFA rejected edited good point"

# @pytest.mark.skip()
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
    print(f"[{test_trace_alignment_single_mock.__name__}] Original bad trace: {mock_bad_trace}")
    print(f"[{test_trace_alignment_single_mock.__name__}] Aligned bad trace: {aligned_trace}")
    assert aligned_accepts, "Automa should accept aligned trace"
    original_rejects = not run_automata(mock_a_dfa_aug, mock_bad_trace)
    assert original_rejects, "Automa should reject original bad trace"

# @pytest.mark.skip()
def test_trace_alignment_mock(mock_a_dfa_aug, mock_dataset):
    _, bp = mock_dataset
    for bad_trace, _ in bp:
        test_trace_alignment_single_mock(mock_a_dfa_aug, bad_trace)
    
@pytest.mark.skip()
def test_trace_disalignment_single_mock(mock_a_dfa_aug, mock_original_trace):
    inv_mock_a_dfa_aug = deepcopy(mock_a_dfa_aug)
    invert_automata(inv_mock_a_dfa_aug)
    good_trace_rejects = not run_automata(inv_mock_a_dfa_aug, mock_original_trace)
    assert good_trace_rejects, "Inverted Automa should reject good trace"
    aligned_trace, _ = trace_alignment(inv_mock_a_dfa_aug, mock_original_trace)
    print(f"[{test_trace_disalignment_single_mock.__name__}] Original trace: {mock_original_trace}")
    print(f"[{test_trace_disalignment_single_mock.__name__}] Aligned original trace: {aligned_trace}")
    aligned_accepts = run_automata(inv_mock_a_dfa_aug, aligned_trace)
    assert aligned_accepts, "Inverted Automa should accetps aligned bad trace"

# @pytest.mark.skip()
def test_trace_disalignment_mock(mock_a_dfa_aug, mock_dataset):
    gp, _ = mock_dataset
    for good_trace, _ in gp:
        test_trace_disalignment_single_mock(mock_a_dfa_aug, good_trace)

#----------TESTS WITH REAL DATA--------------#

# @pytest.mark.skip()
def test_augmented_trace_automata(t_dfa, t_dfa_aug, original_trace, edited_trace):
    t_dfa_accepts = run_automata(t_dfa, original_trace)
    assert t_dfa_accepts, "T_DFA rejected good point"

    t_dfa_aug_accepts = run_automata(t_dfa_aug, edited_trace)
    assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

# @pytest.mark.skip()
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


@pytest.mark.skip()
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

@pytest.mark.skip()
def test_trace_disalignment_single(a_dfa_aug, original_trace):
    inv_mock_a_dfa_aug = deepcopy(a_dfa_aug)
    invert_automata(inv_mock_a_dfa_aug)
    good_trace_rejects = not run_automata(inv_mock_a_dfa_aug, original_trace)
    assert good_trace_rejects, "Inverted Automa should reject good trace"
    aligned_trace, _ = trace_alignment(inv_mock_a_dfa_aug, original_trace)
    print(f"[{test_trace_disalignment_single.__name__}] Original trace: {original_trace}")
    print(f"[{test_trace_disalignment_single.__name__}] Aligned original trace: {aligned_trace}")
    aligned_accepts = run_automata(inv_mock_a_dfa_aug, aligned_trace)
    assert aligned_accepts, "Inverted Automa should accetps aligned bad trace"

@pytest.mark.skip()
def test_trace_disalignment(a_dfa_aug, dataset):
    gp, _ = dataset
    for good_trace, _ in gp:
        test_trace_disalignment_single(a_dfa_aug, good_trace)

#----------GENERAL TESTS--------------#
def test_align():
    # original_trace = [1,2,3,5,6]
    alignment = ("sync_1", "del_2", "add_4", "sync_3", "sync_5", "del_6")
    encoded_alignment = tuple(encode_action_str(a) for a in alignment)
    # print(f"[{test_align.__name__}] encoded_alignment is {encoded_alignment}")
    aligned_trace = align(encoded_alignment)
    # print(f"[{test_align.__name__}] Decoded aligned trace is {aligned_trace}")
    
    correct_alignment = [1,4,3,5]
    assert aligned_trace == correct_alignment, f"""
    Aligned trace is wrong
    corect: {correct_alignment}
    computed: {aligned_trace}
    """


    

    
