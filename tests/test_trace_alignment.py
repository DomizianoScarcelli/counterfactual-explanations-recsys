from recommenders.test import load_dataset
from automata_learning import generate_automata_from_dataset, generate_single_accepting_sequence_dfa, run_automata
from trace_alignment import augment_trace_automata


def test_augmented_trace_automata():
    good_points, bad_points = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    original_trace = good_points[0][0].tolist()
    t_dfa = generate_single_accepting_sequence_dfa(original_trace)
    t_dfa_accepts = run_automata(t_dfa, original_trace)

    repaired_trace = original_trace.copy()
    repaired_trace[10] = f"del_{repaired_trace[10]}"
    repaired_trace[21] = f"del_{repaired_trace[21]}"
    repaired_trace[19] = f"del_{repaired_trace[19]}"
    # repaired_trace[20] = f"add_{repaired_trace[20]}"
    t_dfa_aug = augment_trace_automata(t_dfa)
    t_dfa_aug_accepts = run_automata(t_dfa_aug, repaired_trace)
    # assert t_dfa_accepts and t_dfa_aug_accepts, f"""
    # Automata accepts on original trace? {t_dfa_accepts}
    # Automata accepts on repaired trace? {t_dfa_aug_accepts}
    # Original trace: {original_trace}
    # Repaired trace: {repaired_trace}
    # """
    del_p_assert = t_dfa_accepts and t_dfa_aug_accepts
    repaired_trace = original_trace.copy()
    repaired_trace.insert(10, f"add_100")
    repaired_trace.insert(49, f"add_1456")
    t_dfa_aug_accepts = run_automata(t_dfa_aug, repaired_trace)
    add_p_assert = t_dfa_accepts and t_dfa_aug_accepts
    repaired_trace = original_trace.copy()
    repaired_trace[14] = f"del_{repaired_trace[14]}"
    repaired_trace[48] = f"del_{repaired_trace[48]}"
    repaired_trace.insert(0, f"add_1")
    repaired_trace.insert(23, f"add_14")
    t_dfa_aug_accepts = run_automata(t_dfa_aug, repaired_trace)
    del_add_p_assert = t_dfa_accepts and t_dfa_aug_accepts
    if not del_p_assert:
        print(f"{test_augmented_trace_automata.__name__} del_p failed")
    if not add_p_assert: 
        print(f"{test_augmented_trace_automata.__name__} add_p failed")
    if not del_add_p_assert: 
        print(f"{test_augmented_trace_automata.__name__} add_p + del_p failed")
    if del_p_assert and add_p_assert and del_add_p_assert:
        print(f"{test_augmented_trace_automata.__name__} ALL TESTS PASSED!")


def test_augmented_constraint_automata():
    pass


if __name__ == "__main__":
    test_augmented_trace_automata()
