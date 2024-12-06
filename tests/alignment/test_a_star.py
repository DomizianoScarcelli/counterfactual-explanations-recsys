import pytest
from aalpy.automata.Dfa import Dfa, DfaState

from alignment.a_star import a_star, get_target_states
from alignment.alignment import align
from automata_learning.utils import run_automata


@pytest.fixture()
def mock_dfa():
    states = {sym: DfaState(sym) for sym in ["a", "b", "c", "d", "e",
                                                    "f", "g", "h"]}
    states["a"].transitions["sync_1"] = states["b"]
    states["a"].transitions["sync_3"] = states["g"]
    states["b"].transitions["sync_1"] = states["c"]
    states["b"].transitions["sync_2"] = states["g"]
    states["c"].transitions["sync_1"] = states["d"]
    states["c"].transitions["sync_2"] = states["f"]
    states["c"].transitions["sync_3"] = states["e"]
    states["c"].transitions["sync_4"] = states["e"]
    states["d"].transitions["sync_1"] = states["e"]
    states["e"].transitions["sync_3"] = states["b"]
    states["h"].transitions["sync_2"] = states["d"]
    states["b"].is_accepting = True
    dfa = Dfa(states["a"], list(states.values()))
    dfa.make_input_complete()
    return dfa

def test_GetTargetStates_IsCorrect(mock_dfa: Dfa):
    # Test 1
    leftover_trace = [1, 1, 3]
    states = {s.state_id for s in get_target_states(mock_dfa, leftover_trace)}
    expected = {"c", "d", "e"}
    assert states == expected

    # Test 2
    mock_dfa.reset_to_initial()
    leftover_trace = [1, 3, 3]
    states = {s.state_id for s in get_target_states(mock_dfa, leftover_trace)}
    expected = {"a", "b", "d", "e"}
    assert states == expected


def test_a_star(mock_a_dfa_aug):
    dfa = mock_a_dfa_aug
    target_states = {s for s in dfa.states if s.is_accepting}
    trace = [1,2,4]
    expected = [1,2,4,5]
    alignment = a_star(
        dfa=dfa,
        origin_state=dfa.current_state,
        target_states=target_states,
        remaining_trace=trace,
        leftover_trace_set=set(),
        min_alignment_length=3,
        max_alignment_length=4,
        initial_alignment=tuple(),
        heuristic_fn=lambda _:0, #dijkstra
    )
    assert alignment is not None, "Result is None"
    result = align(alignment)
    assert result == expected, f"Wrong result: {result} != {expected}"

