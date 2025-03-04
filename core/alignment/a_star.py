import heapq
import os
import warnings
from typing import Callable, List, Optional, Sequence, Set, Tuple

from aalpy.automata.Dfa import Dfa, DfaState

if os.environ.get("LINE_PROFILE") == "1":
    pass

from core.alignment.actions import (Action, decode_action, encode_action_str,
                                    is_legal)
from core.alignment.utils import alignment_length, prune_paths_by_length
from config.config import ConfigParams
from exceptions import NoTargetStatesError
from core.heuristics.heuristics import hops
from type_hints import PathInfo, PathsQueue, TraceSplit
from utils.utils import printd


def get_accepting_states(dfa: Dfa, include_sink: bool = True) -> Set[DfaState]:
    if include_sink:
        return {s for s in dfa.states if s.is_accepting}
    return {s for s in dfa.states if s.is_accepting and s.state_id != "sink"}


def get_target_states(dfa: Dfa, leftover_trace: Sequence[int]):
    """
    Identify the set of initial states in a DFA that, after processing the given leftover trace, lead to an accepting state.

    Args:
        dfa: The deterministic finite automaton (DFA) to analyze.  It should
            have states, each with transitions defined as a dictionary of actions
            (e.g., "sync_c").
        leftover_trace: A sequence of integers representing the input trace to
            process. Each integer corresponds to a transition symbol.

    Returns:
        Set[DfaState]: A set of DFA states that can reach an accepting state
            after processing the given `leftover_trace`.

    Example:
        >>> dfa = Dfa(...)  # Initialize a DFA
        >>> leftover_trace = [1, 2, 3]
        >>> target_states = get_target_states(dfa, leftover_trace)
        >>> print([state.state_id for state in target_states])
    """
    accepting_states = get_accepting_states(dfa, include_sink=ConfigParams.INCLUDE_SINK)
    final_states = set()
    for state in dfa.states:
        curr_state = state
        for c in leftover_trace:
            curr_state = curr_state.transitions[c]
        if curr_state in accepting_states:
            final_states.add(state)
    return final_states


def faster_a_star(
    dfa: Dfa,
    trace_split: TraceSplit,
    min_alignment_length: Optional[int],
    max_alignment_length: Optional[int],
):
    """
    Compute an alignment for a given trace using an optimized A* search algorithm, targeting accepting states in the DFA.

    Args:
        dfa (Dfa): The deterministic finite automaton (DFA) used for trace alignment.
                   The DFA should have a defined set of states, each with transitions based on actions.
        trace_split (TraceSplit): A tuple containing three sequences:
            - executed_t: The prefix of the trace already executed in the DFA.
            - alignable_t: The segment of the trace that can be modified during alignment.
            - leftover_t: The suffix of the trace that must remain fixed.
        min_alignment_length (Optional[int]): The minimum allowable length for the alignment.
                                              If `None`, no minimum constraint is enforced.
        max_alignment_length (Optional[int]): The maximum allowable length for the alignment.
                                              If `None`, no maximum constraint is enforced.

    Returns:
        Optional[Tuple[str]]: A tuple of encoded alignment actions if alignment is successful.
                              Returns `None` if no valid alignment is found.

    Raises:
        NoTargetStatesError: If no target (accepting) states are reachable with the given `leftover_t` trace.

    Debug Output:
        If debugging is enabled, prints information about the original trace, the DFA states, and alignment constraints.

    Example:
        >>> dfa = Dfa(...)  # Initialize a DFA
        >>> trace_split = ([1, 2], [3, 4], [5, 6])  # executed, alignable, leftover traces
        >>> alignment = faster_a_star(dfa, trace_split, 2, 10)
        >>> if alignment:
        ...     print("Alignment found:", alignment)
        ... else:
        ...     print("No valid alignment.")
    """
    printd("-----FAST-A*------")

    accepting_states = get_accepting_states(dfa, include_sink=ConfigParams.INCLUDE_SINK)
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
        dfa.step(char)
        initial_alignment.append(char)
    initial_alignment = tuple(initial_alignment)

    target_states = get_target_states(dfa, leftover_t)

    if len(target_states) == 0:
        raise NoTargetStatesError()

    printd(
        f"""
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
          """
    )

    remaining_alignment = a_star(
        dfa=dfa,
        origin_state=dfa.current_state,
        target_states=target_states,
        remaining_trace=alignable_t,
        leftover_trace_set=set(leftover_t),
        min_alignment_length=min_alignment_length,
        max_alignment_length=max_alignment_length,
        initial_alignment=initial_alignment,
        # heuristic_fn=lambda _:0 #dijkstra
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
    leftover_trace_set: Set[int],
    min_alignment_length: Optional[int],
    max_alignment_length: Optional[int],
    heuristic_fn: Optional[Callable] = None,
    initial_alignment: Optional[Tuple[int]] = None,
):
    """
    Performs an A* search to find the optimal alignment of a sequence trace within a deterministic finite automaton (DFA).

    Args:
        dfa (Dfa): The deterministic finite automaton defining the states and transitions.
        origin_state (DfaState): The starting state for the search.
        target_states (Set[DfaState]): The set of accepting (target) states to be reached.
        remaining_trace (List[int]): The part of the trace to be processed during alignment.
        leftover_trace_set (Set[int]): A set of remaining trace elements to be considered for constraints.
        min_alignment_length (Optional[int]): The minimum allowable alignment length. If `None`, no minimum constraint is applied.
        max_alignment_length (Optional[int]): The maximum allowable alignment length. If `None`, no maximum constraint is applied.
        heuristic_fn (Optional[Callable]): A heuristic function for A*. Defaults to the number of hops to the nearest target state if `None`.
        initial_alignment (Optional[Tuple[int]]): Pre-existing alignment actions to be included at the start. Defaults to `None`.

    Returns:
        Optional[Tuple[int]]: The sequence of alignment actions (encoded as integers) that aligns the trace to a target state.
                              Returns `None` if no valid alignment is found.

    Raises:
        None

    Warnings:
        Issues a warning if the `origin_state` or any `target_state` is not present in the DFA.

    Example:
        >>> dfa = Dfa(...)  # Initialize DFA
        >>> origin = dfa.initial_state
        >>> targets = {state for state in dfa.states if state.is_accepting}
        >>> remaining_trace = [1, 2, 3]
        >>> leftover_set = {4, 5}
        >>> alignment = a_star(dfa, origin, targets, remaining_trace, leftover_set, 2, 10)
        >>> if alignment:
        ...     print("Alignment found:", alignment)
        ... else:
        ...     print("No valid alignment.")

    Debugging:
        - Periodically prunes paths if the number of stored paths exceeds a threshold (e.g., 1,000,000).
        - Prints debug information every 1,000 iterations, including steps, number of paths, trace indices, and path costs.

    Details:
        1. **Heuristic Function:**
           The heuristic estimates the cost to reach the nearest target state. Defaults to the number of hops if no `heuristic_fn` is provided.
        2. **Neighbours Generation:**
           Neighbours are filtered based on constraints, legality of actions, and their cost (e.g., `sync` has zero cost).
        3. **Path Management:**
           Uses a priority queue (heap) to explore paths with the lowest cost plus heuristic first.
        4. **Visited States:**
           Tracks visited states to avoid redundant exploration of the same state-action combinations.
        5. **Pruning:**
           Periodically prunes paths to limit memory usage and improve efficiency.

    Implementation Notes:
        - Actions (`sync`, `del`, `add`) are encoded for alignment and decoded as needed.
        - Only valid neighbours satisfying constraints are explored.
        - Paths that reach a target state and satisfy length constraints are returned immediately.
    """
    remaining_trace_idx = len(remaining_trace)

    def heuristic(curr_state, remaining_trace):
        if heuristic_fn:
            return heuristic_fn(curr_state)

        return hops(curr_state, remaining_trace, target_states)

    def get_constrained_neighbours(
        state, curr_char: Optional[int]
    ) -> List[Tuple[DfaState, int, int]]:
        neighbours = []
        inputs_set = set(inputs)
        candidate_states: Set[int] = set(state.transitions)
        if (state.state_id, curr_char) in visited:
            candidate_states -= visited[(state.state_id, curr_char)]
        for action in candidate_states:
            action_type, e = decode_action(action)
            target = state.transitions[action]
            if not is_legal(action, inputs_set, leftover_trace_set):
                continue
            if action_type == Action.SYNC and curr_char is not None:
                if curr_char == e:
                    neighbours.append((target, 0, action))  # sync_e cost = 0
            elif action_type == Action.DEL and curr_char is not None:
                if curr_char == e:
                    cost = 1
                    # if the previous action is an ADD, we have a DEL-ADD combo, which is a REPLACE with cost 1
                    if len(inputs) > 0 and decode_action(inputs[-1])[0] == Action.ADD:
                        cost = 0
                    neighbours.append((target, cost, action))  # del_e cost = 1
            elif action_type == Action.ADD:
                cost = 1
                # if the previous action is a DEL, we have a ADD-DEL combo, which is a REPLACE with cost 1
                if len(inputs) > 0 and decode_action(inputs[-1])[0] == Action.DEL:
                    cost = 0
                neighbours.append((target, cost, action))  # add_e cost = 1

        return neighbours

    invalid_states = [s for s in target_states if s not in dfa.states]
    if origin_state not in dfa.states or invalid_states:
        warnings.warn("Origin or target state not in automaton. Returning None.")
        return None

    paths: PathsQueue = []
    heap_counter = 0
    visited = {}  # {(state_id, current_char): action (int)}

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
        if pbar_counter % 1000 == 0:
            paths = prune_paths_by_length(paths, max_paths=1_000_000)
            printd(f"Steps: {pbar_counter}", level=2)
            printd(f"Num paths: {len(paths)}", level=2)
            printd(f"Remaining trace idx: {remaining_trace_idx}", level=2)
            printd(f"Paths head (20) costs {[p[0] for p in paths[:20]]}", level=2)
            printd(f"Paths tail (20) costs {[p[0] for p in paths[-20:]]}", level=2)

        popped: PathInfo = heapq.heappop(paths)
        cost, heuristic_value, _, current_state, path, inputs, remaining_trace_idx = (
            popped
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

            current_visited_key, current_visited_value = (
                current_state.state_id,
                curr_char,
            ), action
            if (
                current_visited_key in visited
                and current_visited_value in visited[current_visited_key]
            ):
                continue
            if current_visited_key in visited:
                visited[current_visited_key].add(current_visited_value)
            else:
                visited[current_visited_key] = {current_visited_value}

            new_cost = cost + action_cost
            new_path = path + (neighbour,)
            new_inputs = inputs + (action,)

            action_type, _ = decode_action(action)
            new_remaining_trace_idx = (
                remaining_trace_idx - 1
                if action_type in (Action.SYNC, Action.DEL)
                else remaining_trace_idx
            )

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
