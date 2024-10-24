import heapq
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from aalpy.automata.Dfa import Dfa, DfaState
from utils import printd

from alignment.actions import (Action, decode_action, encode_action, encode_action_str)
from alignment.utils import alignment_length, prune_paths_by_length
from heuristics.heuristics import hops
from type_hints import TraceSplit

from exceptions import NoTargetStatesError


def dijkstra(
    dfa: Dfa,
    origin_state: DfaState,
    target_states: Set[DfaState],
    remaining_trace: List[int],
    min_alignment_length: Optional[int],
    max_alignment_length: Optional[int],
    initial_alignment: Optional[Tuple[int]],
):
    def heuristic(_):
        return 0

    return a_star(
        dfa,
        origin_state,
        target_states,
        remaining_trace,
        min_alignment_length,
        max_alignment_length,
        heuristic,
        initial_alignment,
    )


def get_target_states(dfa: Dfa, leftover_trace: Sequence[int]):
    accepting_states = set({state for state in dfa.states if state.is_accepting})
    printd(f"Accepting states: {[s.state_id for s in accepting_states]}")
    final_states = set()
    for state in dfa.states:
        curr_state = state
        for c in leftover_trace:
            new_state = curr_state.transitions[f"sync_{c}"]
            curr_state = new_state
        if curr_state in accepting_states:
            final_states.add(state)
    return final_states


def faster_dijkstra(
    dfa: Dfa,
    trace_split: TraceSplit,
    min_alignment_length: Optional[int],
    max_alignment_length: Optional[int],
):
    printd("-----FAST-DIJKSTRA------")

    accepting_states = set(s for s in dfa.states if s.is_accepting)
    executed_t, alignable_t, leftover_t = trace_split

    # Checks
    if len(executed_t) > 0:
        assert isinstance(
            executed_t[0], int
        ), f"Executed trace isn't a sequence of int, but {isinstance(executed_t[0], int)}"
    if len(alignable_t) > 0:
        assert isinstance(
            alignable_t[0], int
        ), f"Alignable trace isn't a sequence of int, but {isinstance(alignable_t[0], int)}"
    if len(leftover_t) > 0:
        assert isinstance(
            leftover_t[0], int
        ), f"Leftover trace isn't a sequence of int, but {isinstance(leftover_t[0], int)}"

    # Execute the "executed_t" trace
    initial_alignment = []
    for char in executed_t:
        char = f"sync_{char}"
        dfa.step(char)
        encoded_char = encode_action_str(char)
        initial_alignment.append(encoded_char)
    initial_alignment = tuple(initial_alignment)

    # TODO: Test if target states are actually computed correctly
    target_states = get_target_states(dfa, leftover_t)


    if len(target_states) == 0:
        raise NoTargetStatesError()
    
    # # Adjust min and max alignment length according to the leftover trace length
    # if min_alignment_length and leftover_t:
    #     min_alignment_length -= len(leftover_t)
    # if max_alignment_length and leftover_t:
    #     max_alignment_length -= len(leftover_t)

    printd(f"""
          ---DEBUG---
          Original trace: {executed_t + alignable_t + leftover_t}
          Executed trace: {executed_t}
          Mutable trace: {alignable_t}
          Fixed end trace: {leftover_t}
          ---
          Initial state: {dfa.initial_state.state_id}
          State after executed trace: {dfa.current_state.state_id}
          Accepting sates: {[s.state_id for s in accepting_states]}
          Target states: {[s.state_id for s in target_states]}
          ---
          (Min, Max) Alignment Length: ({min_alignment_length}, {max_alignment_length})
          """)

    remaining_alignment = a_star(
        dfa=dfa,
        origin_state=dfa.current_state,
        target_states=target_states,
        remaining_trace=alignable_t,
        min_alignment_length=min_alignment_length,
        max_alignment_length=max_alignment_length,
        initial_alignment=initial_alignment,
    )
    if remaining_alignment:
        return remaining_alignment + tuple(
            encode_action_str(action) for action in [f"sync_{c}" for c in leftover_t]
        )
    return None


def a_star(
    dfa: Dfa,
    origin_state: DfaState,
    target_states: Set[DfaState],
    remaining_trace: List[int],
    min_alignment_length: Optional[int],
    max_alignment_length: Optional[int],
    heuristic_fn: Optional[Callable] = None,
    initial_alignment: Optional[Tuple[int]] = None,
):
    remaining_trace_idx = len(remaining_trace)

    def heuristic(curr_state, remaining_trace):
        if heuristic_fn:
            return heuristic_fn(curr_state)

        return hops(curr_state, remaining_trace, target_states)

    def get_constrained_neighbours(state, curr_char: Optional[int]):
        neighbours = []
        for p, target in state.transitions.items():
            encoded_p = encode_action_str(p) if isinstance(p, str) else p
            action_type, e = decode_action(encoded_p)
            if action_type == Action.SYNC and curr_char is not None:
                add_e = encode_action(Action.ADD, e)
                del_e = encode_action(Action.DEL, e)
                already_added = p in inputs or add_e in inputs or del_e in inputs
                if curr_char == e and not already_added:
                    neighbours.append((target, 0, encoded_p))  # sync_e cost = 0
            elif action_type == Action.DEL and curr_char is not None:
                sync_e = encode_action(Action.SYNC, e)
                add_e = encode_action(Action.ADD, e)
                already_added = p in inputs or sync_e in inputs or add_e in inputs
                if curr_char == e and not already_added:
                    neighbours.append((target, 1, encoded_p))  # del_e cost = 1
            elif action_type == Action.ADD:
                sync_e = encode_action(Action.SYNC, e)
                del_e = encode_action(Action.DEL, e)
                already_added = p in inputs or sync_e in inputs or del_e in inputs
                if not already_added:
                    neighbours.append((target, 1, encoded_p))  # add_e cost = 1

        return neighbours

    invalid_states = [s for s in target_states if s not in dfa.states]
    if origin_state not in dfa.states or invalid_states:
        warnings.warn("Origin or target state not in automaton. Returning None.")
        return None

    paths = []
    heap_counter = 0
    visited = set()

    if initial_alignment:
        heapq.heappush(
            paths,
            (
                0,
                0,
                heap_counter,
                origin_state,
                (origin_state,),
                initial_alignment,
                remaining_trace_idx,
            ),
        )
    else:
        heapq.heappush(
            paths,
            (
                0,
                0,
                heap_counter,
                origin_state,
                (origin_state,),
                (),
                remaining_trace_idx,
            ),
        )

    pbar_counter = 0
    while paths:
        pbar_counter += 1
        if pbar_counter % 1000 == 0 :
            paths = prune_paths_by_length(paths, max_paths=1_000_000)
            printd(f"Steps: {pbar_counter}", level=2)
            printd(f"Num paths: {len(paths)}", level=2)
            printd(f"Remaining trace idx: {remaining_trace_idx}", level=2)
            printd(f"Paths head (20) costs {[p[0] for p in paths[:20]]}", level=2)
            printd(f"Paths tail (20) costs {[p[0] for p in paths[-20:]]}", level=2)

        cost, heuristic_value, _, current_state, path, inputs, remaining_trace_idx = heapq.heappop(
            paths
        )

        curr_alignment_length = alignment_length(inputs)
        if current_state in target_states and remaining_trace_idx == 0:
            if (
                min_alignment_length is None
                or curr_alignment_length >= min_alignment_length
            ) and (
                max_alignment_length is None
                or curr_alignment_length <= max_alignment_length
            ):
                return tuple(inputs)

        curr_char = (
            remaining_trace[-remaining_trace_idx] if remaining_trace_idx > 0 else None
        )
        neighbours = get_constrained_neighbours(current_state, curr_char)

        for neighbour, action_cost, action in neighbours:
            if action in set(inputs):
                continue

            current_visited = (current_state.state_id, curr_char, action)
            # if decode_action(action)[0] != Action.SYNC:
            if current_visited in visited:
                continue
            visited.add(current_visited)

            new_cost = cost + action_cost
            new_path = path + (neighbour,)
            new_inputs = inputs + (action,)

            new_remaining_trace_idx = remaining_trace_idx
            action_type, _ = decode_action(action)
            if action_type in {Action.SYNC, Action.DEL}:
                new_remaining_trace_idx -= 1

            heap_counter += 1
            heuristic_value = heuristic(current_state, new_inputs)
            heapq.heappush(
                paths,
                (
                    new_cost,
                    heuristic_value,
                    heap_counter,
                    neighbour,
                    new_path,
                    new_inputs,
                    new_remaining_trace_idx,
                ),
            )

    return None
