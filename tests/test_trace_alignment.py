import pytest
from recommenders.test import load_dataset
from automata_learning import generate_automata, generate_automata_from_dataset, generate_single_accepting_sequence_dfa, run_automata, NumItems
from trace_alignment import augment_constraint_automata, augment_trace_automata, create_intersection_automata, run_trace_alignment, _deprecated_create_intersection_automata
from test_automata_learning import mock_dataset

#----------TESTS WITH MOCK DATA--------------#
@pytest.fixture
def mock_original_trace(mock_dataset):
    gp, _ = mock_dataset
    return gp[0][0].tolist()

@pytest.fixture
def mock_bad_trace(mock_dataset):
    _, bp = mock_dataset
    return bp[1][0].tolist()

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

def test_augmented_trace_automata_mock(mock_t_dfa, mock_t_dfa_aug, mock_original_trace, mock_edited_trace):
    # mock_t_dfa.visualize("mock_t_dfa")
    # mock_t_dfa_aug.visualize("mock_t_dfa_aug")

    t_dfa_accepts = run_automata(mock_t_dfa, mock_original_trace)
    assert t_dfa_accepts, "T_DFA rejected good point"

    t_dfa_aug_accepts = run_automata(mock_t_dfa_aug, mock_edited_trace)
    assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

def test_augmented_constraint_automata_mock(mock_a_dfa, mock_a_dfa_aug, mock_original_trace, mock_edited_trace): 
    # mock_a_dfa.visualize("mock_a_dfa")
    # mock_a_dfa_aug.visualize("mock_a_dfa_aug")

    a_dfa_accepts = run_automata(mock_a_dfa, mock_original_trace)
    assert a_dfa_accepts, "A_DFA rejected good point"

    a_dfa_accepts = run_automata(mock_a_dfa_aug, mock_edited_trace)
    assert a_dfa_accepts, "A_DFA rejected edited good point"

def test_run_trace_alignment_bad_trace_mock(mock_a_dfa_aug, mock_bad_trace):
    print(f"Bad trace is: ", mock_bad_trace)
    a_dfa_aug_accepts = run_automata(mock_a_dfa_aug, mock_bad_trace)
    assert not a_dfa_aug_accepts, f"Bad trace should be accepted"
    
    # TODO: if a character cannot be read by the automata, everything
    # collapses, see how to handle this
    alignment, cost = run_trace_alignment(mock_a_dfa_aug, mock_bad_trace)
    print(f"Best alignment {alignment} with cost {cost}")




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
def t_dfa_aug(t_dfa):
    return augment_trace_automata(t_dfa)

@pytest.fixture
def a_dfa_aug(a_dfa, t_dfa):
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

# @pytest.mark.skip()
# def test_get_shortest_alignment_dijkstra(a_dfa_aug):
#     pass

@pytest.mark.skip()
def test_run_trace_alignment_good_trace(a_dfa_aug, original_trace):
    a_dfa_aug_accepts = run_automata(a_dfa_aug, original_trace)
    assert a_dfa_aug_accepts, f"Original trace should be accepted"
    
    # NOTE: I don't know how to modify the original trace in order to not be
    # accepted, and to not use a bad trace
    original_trace[10] = 1024
    a_dfa_aug_accepts = run_automata(a_dfa_aug, original_trace)
    assert not a_dfa_aug_accepts, f"Modified trace shouldn't be accepted"

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




    

    
