import heapq
from aalpy.automata.Dfa import Dfa, DfaState
from typing import List, Optional, Set, Callable
import warnings
from memory_profiler import memory_usage
import tracemalloc

def log_memory_usage(message: str):
    mem = memory_usage(-1, interval=0.1, timeout=1)
    print(f"{message}: Current memory usage: {mem[0]:.4f} MB")

def analyze_tracemalloc():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[Top 10 Memory Allocations]")
    for stat in top_stats[:10]:
        print(stat)

def get_shortest_alignment_dijkstra(dfa: Dfa, 
                                    origin_state: DfaState,
                                    target_states: Set[DfaState],
                                    remaining_trace: List[int],
                                    min_alignment_length: Optional[int],
                                    max_alignment_length: Optional[int]):
    heuristic = lambda _: 0
    return get_shortest_alignment_a_star(dfa, origin_state, target_states, remaining_trace, min_alignment_length, max_alignment_length, heuristic)

def get_shortest_alignment_a_star(dfa: Dfa, 
                                  origin_state: DfaState,
                                  target_states: Set[DfaState],
                                  remaining_trace: List[int],
                                  min_alignment_length: Optional[int],
                                  max_alignment_length: Optional[int],
                                  heuristic_fn: Optional[Callable] = None):
    remaining_trace_idx = len(remaining_trace)
    tracemalloc.start()
    
    def heuristic(curr_inputs):
        if heuristic_fn:
            return heuristic_fn(curr_inputs)
        ALPHA = 1
        max_steps = max_alignment_length or 50
        return abs(max_steps - alignment_length(curr_inputs)) * ALPHA

    def get_constrained_neighbours(state, curr_char: Optional[int]):
        neighbours = []
        for p, target in state.transitions.items():
            if (state.state_id, p) in visited:
                continue

            if "sync" in p and curr_char is not None:
                extracted_p = int(p.replace("sync_", ""))
                already_added = f"sync_{extracted_p}" in set(inputs) or f"add_{extracted_p}" in set(inputs)
                if curr_char == extracted_p and not already_added:
                    neighbours.append((target, 0, p))  # sync_e cost = 0
            if "del" in p and curr_char is not None:
                extracted_p = int(p.replace("del_", ""))
                if curr_char == extracted_p:
                    neighbours.append((target, 1, p))  # del_e cost = 1
            if "add" in p:
                extracted_p = int(p.replace("add_", ""))
                already_added = f"sync_{extracted_p}" in set(inputs) or f"add_{extracted_p}" in set(inputs)
                if not already_added:
                    neighbours.append((target, 1, p))  # add_e cost = 1

        return neighbours

    def alignment_length(curr_alignment):
        return sum(1 if "sync" in c or "add" in c else 0 for c in curr_alignment)
    
    invalid_states = [s for s in target_states if not s in dfa.states]
    if origin_state not in dfa.states or invalid_states:
        warnings.warn('Origin or target state not in automaton. Returning None.')
        return None

    visited = set()
    paths = []
    heap_counter = 0

    # Modify to add heuristic to the cost calculation
    heapq.heappush(paths, (0, heap_counter, origin_state, (origin_state,), (), remaining_trace_idx, visited))
    
    pbar_counter = 0
    while paths:
        pbar_counter += 1
        if pbar_counter % 1000 == 0:
            print(f"Steps: {pbar_counter}")
            print(f"Remaining paths: {len(paths)}")
            print(f"Remaining trace idx: {remaining_trace_idx}")
            print(f"Remaining trace: {remaining_trace[-remaining_trace_idx:]}")
            log_memory_usage("")
            analyze_tracemalloc()

        cost, _, current_state, path, inputs, remaining_trace_idx, visited = heapq.heappop(paths)

        curr_alignment_length = alignment_length(inputs)
        if current_state in target_states and remaining_trace_idx == 0:
            if min_alignment_length is None or curr_alignment_length >= min_alignment_length:
                if max_alignment_length is None or curr_alignment_length <= max_alignment_length:
                    return tuple(inputs)
        
        curr_char = remaining_trace[-remaining_trace_idx] if remaining_trace_idx > 0 else None
        neighbours = get_constrained_neighbours(current_state, curr_char)

        for neighbour, action_cost, action in neighbours:
            new_cost = cost + action_cost
            new_path = path + (neighbour,)
            new_inputs = inputs + (action,)

            new_visited = visited.copy()
            if (current_state.state_id, action) in new_visited:
                continue

            new_visited.add((current_state.state_id, action))

            new_remaining_trace_idx = remaining_trace_idx
            if "sync" in action or "del" in action:
                new_remaining_trace_idx -= 1

            heap_counter += 1
            # Include heuristic cost in the priority queue
            priority = new_cost + heuristic(curr_inputs=inputs)
            heapq.heappush(paths, (priority, heap_counter, neighbour, new_path, new_inputs, new_remaining_trace_idx, new_visited))

    return None

