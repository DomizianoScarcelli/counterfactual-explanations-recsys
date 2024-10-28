import pytest
from alignment.a_star import a_star, a_star_parallel
from alignment.alignment import align
import time

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
    )
    assert alignment is not None, "Result is None"
    result = align(alignment)
    assert result == expected, f"Wrong result: {result} != {expected}"


def test_a_star_parallel(mock_a_dfa_aug):
    dfa = mock_a_dfa_aug
    target_states = {s for s in dfa.states if s.is_accepting}
    trace = [1,2,4]
    expected = [1,2,4,5]
    alignment = a_star_parallel(
        dfa=dfa,
        origin_state=dfa.current_state,
        target_states=target_states,
        remaining_trace=trace,
        leftover_trace_set=set(),
        min_alignment_length=3,
        max_alignment_length=4,
        initial_alignment=tuple(),
    )
    assert alignment is not None, "Result is None"
    result = align(alignment)
    assert result == expected, f"Wrong result: {result} != {expected}"

def test_a_star_parallel_timing(mock_a_dfa_aug):
    start = time.time()
    for _ in range(1000):
        test_a_star(mock_a_dfa_aug)
    end = time.time()
    a_star_time = end - start
    start = time.time()
    for _ in range(1000):
        test_a_star_parallel(mock_a_dfa_aug)
    end = time.time()
    a_star_parallel_time = end - start
    print(f"A star timing: {a_star_time}")
    print(f"Parallel A star timing: {a_star_parallel_time}")
