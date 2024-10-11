import pytest
from recommenders.test import load_dataset
from automata_learning import (generate_automata, 
                               generate_automata_from_dataset, 
                               generate_single_accepting_sequence_dfa, 
                               run_automata, 
                               NumItems)
from trace_alignment import (augment_constraint_automata, 
                             augment_trace_automata, 
                             create_intersection_automata, 
                             get_shortest_alignment_dijkstra, planning_aut_to_constraint_aut, 
                             run_trace_alignment, 
                             _deprecated_create_intersection_automata, 
                             constraint_aut_to_planning_aut,
                             align)
import torch

#----------TESTS WITH MOCK DATA--------------#
@pytest.fixture
def mock_dataset():
    """
    A tiny and controllable set of good and bad points in order to visualize
    the learned automata and do further debug

    Assume the unviverse of all possible point (whic may not be contained in
                                                the dataset) is [1,2,3,4,5,6].
    This is equivalent to the universe of items in the recommender system real
    example
    """
    gp =[(torch.tensor([1,3,2]),True),
         (torch.tensor([1,2,3]),True),
         (torch.tensor([2,3,1,5]),True),
         (torch.tensor([5,4,1]),True)]

    bp =[(torch.tensor([1,2,4]),False),
         (torch.tensor([1,3,2,5]),False),
         (torch.tensor([1,2,5]),False), 
         (torch.tensor([5,3,4,1]),False), 
         (torch.tensor([2,3,5,1]),False)]
    return (gp, bp)

# TODO: see how it's possible to combine mock and real fixtures/tests in order
# to not write a lot of repeated code
# (https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures)
@pytest.fixture
def mock_original_trace(mock_dataset):
    gp, _ = mock_dataset
    return gp[0][0].tolist()

@pytest.fixture
def mock_bad_trace(mock_dataset):
    _, bp = mock_dataset
    return bp[3][0].tolist()

@pytest.fixture
def mock_edited_trace(mock_dataset):
    gp, _ = mock_dataset
    original_trace = gp[0][0].tolist()
    
    # Since we will be doing del_at, add_ot, we can only consider another good
    # trace where ot doesn't appear in the original trace
    trace_chars = set(original_trace)
    for another_trace, _ in gp:
        if len(set(another_trace) & trace_chars) == 0:
            break
    # [2720, 365, 1634, 1229, 140, 3298, 1664, 160, 1534, 1233, 618, 267, 2490, 2492, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 213, 2827, 532, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 23, 415, 1988, 694, 133, 1536, 510, 290, 152, 204, 1770, 1273, 289, 462, 165]
    # [2720, 365, 1634, 1229, 140, 351, 1664, 160, 1534, 1233, 618, 267, 2490, 213, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 2492, 2827, 269, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 1988, 415, 23, 694, 133, 1536, 510, 290, 152, 204, 1034, 1273, 289, 462, 165]
    test_trace = []
    for (oc, ac) in zip(original_trace, another_trace):
        if oc == ac:
            test_trace.append(oc)
        else:
            test_trace.extend([f"del_{oc}", f"add_{ac}"])
    return test_trace

@pytest.fixture
def mock_t_dfa(mock_original_trace):
    return generate_single_accepting_sequence_dfa(mock_original_trace)

@pytest.fixture
def mock_a_dfa(mock_dataset):
    return generate_automata_from_dataset(mock_dataset)

@pytest.fixture
def mock_t_dfa_aug(mock_original_trace):
    # Generate another t_dfa since it's augmented on place
    t_dfa = generate_single_accepting_sequence_dfa(mock_original_trace)
    return augment_trace_automata(t_dfa, num_items=NumItems.MOCK)

@pytest.fixture
def mock_a_dfa_aug(mock_dataset, mock_t_dfa):
    a_dfa = generate_automata_from_dataset(mock_dataset, load_if_exists=False)
    return augment_constraint_automata(a_dfa, mock_t_dfa)

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
    alignment, cost = run_trace_alignment(mock_a_dfa_aug, mock_bad_trace)
    print(f"Best alignment {alignment} with cost {cost}")

@pytest.mark.skip()
def test_get_shortest_alignment_dijkstra_mock(mock_a_dfa_aug, mock_bad_trace):
    constraint_aut_to_planning_aut(mock_a_dfa_aug)
    # run_automata(mock_a_dfa_aug, [f"sync_{e}" for e in mock_bad_trace])
    remaining_trace = list(reversed(mock_bad_trace)).copy()
    added_symbols = set()
    final_state = [s for s in mock_a_dfa_aug.states if s.is_accepting][0]
    print(f"Final state is:", final_state.state_id)
    path = get_shortest_alignment_dijkstra(dfa=mock_a_dfa_aug, 
                                    origin_state=mock_a_dfa_aug.initial_state, 
                                    target_state=final_state,
                                    remaining_trace=remaining_trace,
                                    added_symbols=added_symbols)
    assert path is not None, "No best path found"
    print("best path is: ", path)
    if bad_trace == [5,3,4,1]:
        expected_path = ("sync_5", "del_3", "sync_4", "sync_1")
        assert path == expected_path ,f"""Best path is not the correct one:
            {expected_path} != {path}""" 
    
    planning_aut_to_constraint_aut(mock_a_dfa_aug)
    aligned_trace = align(mock_bad_trace, path)
    aligned_accepts = run_automata(mock_a_dfa_aug, aligned_trace)
    assert aligned_accepts, "Automa should accept aligned trace"
    original_rejects = not run_automata(mock_a_dfa_aug, mock_bad_trace)
    assert original_rejects, "Automa should reject original bad trace"

def test_get_shortest_alignment_dijkstra_bad_dataset_mock(mock_a_dfa_aug, mock_dataset):
    _, bp = mock_dataset
    for bad_trace, _ in bp:
        test_get_shortest_alignment_dijkstra_mock(mock_a_dfa_aug, bad_trace)
    

#----------TESTS WITH REAL DATA--------------#
@pytest.fixture
def dataset():
    g, b = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    new_g, new_b = [], []
    ids = set()
    for p, l in g:
        if tuple(p.tolist()) in ids:
            continue
        new_g.append((p,l))
        ids.add(tuple(p.tolist()))
    for p, l in b:
        if tuple(p.tolist()) in ids:
            continue
        new_b.append((p,l))
        ids.add(tuple(p.tolist()))

    dataset = (new_g, new_b)
    return dataset

@pytest.fixture
def original_trace(dataset):
    gp, _ = dataset
    return gp[0][0].tolist()

@pytest.fixture
def bad_trace(dataset):
    _, bp = dataset
    return bp[1][0].tolist()

@pytest.fixture
def edited_trace(dataset):
    gp, _ = dataset
    original_trace = gp[0][0].tolist()
    
    # Since we will be doing del_at, add_ot, we can only consider another good
    # trace where ot doesn't appear in the original trace
    trace_chars = set(original_trace)
    for another_trace, _ in gp:
        if len(set(another_trace) & trace_chars) == 0:
            break
    # [2720, 365, 1634, 1229, 140, 3298, 1664, 160, 1534, 1233, 618, 267, 2490, 2492, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 213, 2827, 532, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 23, 415, 1988, 694, 133, 1536, 510, 290, 152, 204, 1770, 1273, 289, 462, 165]
    # [2720, 365, 1634, 1229, 140, 351, 1664, 160, 1534, 1233, 618, 267, 2490, 213, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 2492, 2827, 269, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 1988, 415, 23, 694, 133, 1536, 510, 290, 152, 204, 1034, 1273, 289, 462, 165]
    test_trace = []
    for (oc, ac) in zip(original_trace, another_trace):
        if oc == ac:
            test_trace.append(oc)
        else:
            test_trace.extend([f"del_{oc}", f"add_{ac}"])
    return test_trace

@pytest.fixture
def t_dfa(original_trace):
    return generate_single_accepting_sequence_dfa(original_trace)

@pytest.fixture
def a_dfa(dataset):
    return generate_automata_from_dataset(dataset)

@pytest.fixture
def t_dfa_aug(original_trace):
    t_dfa = generate_single_accepting_sequence_dfa(original_trace)
    return augment_trace_automata(t_dfa)

@pytest.fixture
def a_dfa_aug(dataset, t_dfa):
    a_dfa = generate_automata_from_dataset(dataset, load_if_exists=True)
    return augment_constraint_automata(a_dfa, t_dfa)

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


@pytest.mark.skip()
def test_get_shortest_alignment_dijkstra(a_dfa_aug, bad_trace):
    # Bad trace = [5,3,4,1]
    constraint_aut_to_planning_aut(a_dfa_aug)
    # run_automata(mock_a_dfa_aug, [f"sync_{e}" for e in mock_bad_trace])
    remaining_trace = list(reversed(bad_trace)).copy()
    added_symbols = set()
    final_state = [s for s in a_dfa_aug.states if s.is_accepting][0]
    print(f"Final state is:", final_state.state_id)
    path = get_shortest_alignment_dijkstra(dfa=a_dfa_aug, 
                                    origin_state=a_dfa_aug.initial_state, 
                                    target_state=final_state,
                                    remaining_trace=remaining_trace,
                                    added_symbols=added_symbols)
    assert path is not None, "No best path found"
    print("best path is: ", path)
    best_action = path[0]
    assert best_action == "sync_1", f"Best action is not the correct one: 1 != {best_action}"
    # remaining_trace.pop()

@pytest.mark.skip()
def test_run_trace_alignment_good_trace(a_dfa_aug, original_trace):
    a_dfa_aug_accepts = run_automata(a_dfa_aug, original_trace)
    assert a_dfa_aug_accepts, f"Original trace should be accepted"
    
    # NOTE: I don't know how to modify the original trace in order to not be
    # accepted, and to not use a bad trace
    original_trace[10] = 1024
    # a_dfa_aug_accepts = run_automata(a_dfa_aug, original_trace)
    # assert not a_dfa_aug_accepts, f"Modified trace shouldn't be accepted"

    alignment, cost = run_trace_alignment(a_dfa_aug, original_trace)
    print(f"Best alignment {alignment} with cost {cost}")
    assert cost == 4

@pytest.mark.skip()
def test_run_trace_alignment_bad_trace(a_dfa_aug, bad_trace):
    print(f"Bad trace is: ", bad_trace)
    a_dfa_aug_accepts = run_automata(a_dfa_aug, bad_trace)
    assert not a_dfa_aug_accepts, f"Bad trace should be accepted"
    
    # TODO: if a character cannot be read by the automata, everything
    # collapses, see how to handle this
    alignment, cost = run_trace_alignment(a_dfa_aug, bad_trace)
    print(f"Best alignment {alignment} with cost {cost}")

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


    

    
