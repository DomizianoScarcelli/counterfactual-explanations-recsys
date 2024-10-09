import pytest
from recommenders.test import load_dataset
from automata_learning import generate_automata_from_dataset, generate_single_accepting_sequence_dfa, run_automata
from trace_alignment import augment_constraint_automata, augment_trace_automata

def test_augmented_trace_automata():
    good_points, bad_points = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    original_trace = good_points[0][0].tolist()
    t_dfa = generate_single_accepting_sequence_dfa(original_trace)
    t_dfa_accepts = run_automata(t_dfa, original_trace)

    repaired_trace = original_trace.copy()
    repaired_trace[10] = f"del_{repaired_trace[10]}"
    repaired_trace[21] = f"del_{repaired_trace[21]}"
    repaired_trace[19] = f"del_{repaired_trace[19]}"
    t_dfa_aug = augment_trace_automata(t_dfa)
    t_dfa_aug_accepts = run_automata(t_dfa_aug, repaired_trace)
    del_p_assert = t_dfa_accepts and t_dfa_aug_accepts
    assert del_p_assert
    repaired_trace = original_trace.copy()
    repaired_trace.insert(10, f"add_100")
    repaired_trace.insert(49, f"add_1456")
    t_dfa_aug_accepts = run_automata(t_dfa_aug, repaired_trace)
    add_p_assert = t_dfa_accepts and t_dfa_aug_accepts
    assert add_p_assert
    repaired_trace = original_trace.copy()
    repaired_trace[14] = f"del_{repaired_trace[14]}"
    repaired_trace[48] = f"del_{repaired_trace[48]}"
    repaired_trace.insert(0, f"add_1")
    repaired_trace.insert(23, f"add_14")
    t_dfa_aug_accepts = run_automata(t_dfa_aug, repaired_trace)
    del_add_p_assert = t_dfa_accepts and t_dfa_aug_accepts
    assert del_add_p_assert

def test_augmented_constraint_automata(): 
    dataset = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    gp, bp = dataset
    original_trace = gp[0][0].tolist()
    t_dfa = generate_single_accepting_sequence_dfa(original_trace)
    a_dfa = generate_automata_from_dataset(dataset)
    a_dfa_aug = augment_constraint_automata(a_dfa, t_dfa)

    # Test if automata is built correctly
    # a_dfa should accept points from the good_points and reject points from bad_points
    gp_1 = gp[5][0].tolist()
    a_dfa_accepts = run_automata(a_dfa, gp_1)
    assert a_dfa_accepts, "A_DFA rejected good point"
    # a_dfa_aug should accept points from good_points that have been edited
    # with add_p and del_p propositions, and rejecting bad_points that have
    # been edited with add_p and del_p propositions
    gp_1_edit = gp_1.copy()
    gp_1_edit[10] = f"del_{gp_1_edit[10]}"
    a_dfa_accepts = run_automata(a_dfa_aug, gp_1_edit)
    assert a_dfa_accepts, "A_DFA rejected good point edited with del_p proposition"
    gp_1_edit = gp_1.copy()
    gp_1_edit[20] = f"add_{gp_1_edit[20]}"
    a_dfa_accepts = run_automata(a_dfa_aug, gp_1_edit)
    a = run_automata(a_dfa, gp_1_edit)
    assert a_dfa_accepts, "A_DFA rejected good point edited with add_p proposition"
    print(a)



