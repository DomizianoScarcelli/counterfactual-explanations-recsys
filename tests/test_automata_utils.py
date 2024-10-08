import pytest
from aalpy.automata.Dfa import Dfa
from recommenders.test import load_dataset
from automata_learning import generate_automata_from_dataset, generate_single_accepting_sequence_dfa
from automata_utils import has_path_to_accepting_state

def test_has_path_to_accepting_state():
    good_points, bad_points = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    good_point = good_points[0][0].tolist()
    a_dfa = generate_automata_from_dataset((good_points, bad_points))
    t_dfa = generate_single_accepting_sequence_dfa(good_point)
    a_dfa_accepting_1 = has_path_to_accepting_state(a_dfa, [good_point[0]])
    a_dfa_accepting_2 = has_path_to_accepting_state(a_dfa, good_point[:10])

    #TODO: this is inheritely wrong, since a bad point can lead the automaton
    # to a state that can actually reach an accepting state if the correct
    # chars are read. Those char won't be in the bad point, but this won't be
    # notices by the has_path_to_accepting_state algorithm, since it just looks
    # if a state can actually end up into a final state.

    bad_point = bad_points[0][0].tolist()
    a_dfa_rejecting = not has_path_to_accepting_state(a_dfa, bad_point)
    assert a_dfa_accepting_1 and a_dfa_accepting_2 and a_dfa_rejecting, f"""
    a_dfa_accepting_1 should be True, {a_dfa_accepting_1}
    a_dfa_accepting_2 should be True, {a_dfa_accepting_2}
    a_dfa_rejecting should be True, {a_dfa_rejecting}
    """

