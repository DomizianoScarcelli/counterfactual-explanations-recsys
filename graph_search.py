import heapq
from aalpy.automata.Dfa import Dfa, DfaState
from typing import List, Optional
import warnings

def get_shortest_alignment_dijkstra(dfa: Dfa, 
                                    origin_state: DfaState,
                                    target_state: DfaState,
                                    remaining_trace: List[int],
                                    min_alignment_length: int):
    remaining_trace = remaining_trace.copy()
    
    def get_constrained_neighbours(state, curr_char: Optional[int]):
        neighbours = []
        for p, target in state.transitions.items():
            if (state.state_id, p) in visited:
                continue

            if "sync" in p and curr_char is not None:
                extracted_p = int(p.replace("sync_", ""))
                if curr_char == extracted_p:
                    neighbours.append((target, 0, p))  # sync_e cost = 0
            if "del" in p and curr_char is not None:
                extracted_p = int(p.replace("del_", ""))
                if curr_char == extracted_p:
                    neighbours.append((target, 1, p))  # sync_e cost = 0
            if "add" in p:
                extracted_p = int(p.replace("add_", ""))
                if f"sync_{extracted_p}" not in set(inputs) and f"add_{extracted_p}" not in set(inputs):
                    neighbours.append((target, 1, p))  # add_e cost = 1

        return neighbours

    def alignment_length(curr_alignment):
        return sum(1 if "sync" in c or "add" in c else -1 for c in curr_alignment)

    if origin_state not in dfa.states or target_state not in dfa.states:
        warnings.warn('Origin or target state not in automaton. Returning None.')
        return None

    # Set to track visited (state_id, action) pairs to avoid infinite loops
    visited = set()
    
    # Priority queue to track (cost, state, path, inputs, remaining_trace)
    paths = []
    heap_counter = 0 #in order to avoid comparison of states when doing heappop
    heapq.heappush(paths, (0, heap_counter, origin_state, [origin_state], [], remaining_trace, visited))
    
    pbar_counter = 0
    while paths:
        pbar_counter += 1
        if pbar_counter % 1000 == 0:
            print(f"Steps: {pbar_counter}")
            print(f"Remaining paths: {len(paths)}")
            print(f"Remaining trace: {len(remaining_trace)}")
        # Get the path with the lowest cost
        cost, _, current_state, path, inputs, remaining_trace, visited = heapq.heappop(paths)

        curr_alignment_length = alignment_length(inputs)
        if current_state == target_state and not remaining_trace and curr_alignment_length >= min_alignment_length:
            return tuple(inputs)

        if remaining_trace:
            curr_char = remaining_trace[-1]
            neighbours = get_constrained_neighbours(current_state, curr_char)
        else:
            neighbours = get_constrained_neighbours(current_state, None)

        for neighbour, action_cost, action in neighbours:
            new_cost = cost + action_cost
            new_path = path + [neighbour]
            new_inputs = inputs + [action]
            
            new_visited = visited.copy()
            if (current_state.state_id, action) in new_visited:
                continue
            
            # Mark this (state, action) pair as visited
            new_visited.add((current_state.state_id, action))
            
            #TODO: Maybe I don't need the copy, so I can avoid to store it into the heapq
            new_remaining_trace = remaining_trace.copy()
            if "sync" in action or "del" in action:
                new_remaining_trace.pop()
            
            # Push the new path into the priority queue
            heap_counter += 1
            heapq.heappush(paths, (new_cost, heap_counter, neighbour, new_path, new_inputs, new_remaining_trace, new_visited))

def get_shortest_alignment_a_star(dfa: Dfa, 
                                  origin_state: DfaState,
                                  target_state: DfaState,
                                  remaining_trace: List[int],
                                  min_alignment_length: int,
                                  max_steps: int = 10000):
    remaining_trace = remaining_trace.copy()

    def heuristic(curr_inputs):
        ALPHA = 1
        return abs(max_steps - alignment_length(curr_inputs)) * ALPHA

    def get_constrained_neighbours(state, curr_char: Optional[int]):
        neighbours = []
        for p, target in state.transitions.items():
            if (state.state_id, p) in visited:
                continue

            if "sync" in p and curr_char is not None:
                extracted_p = int(p.replace("sync_", ""))
                if curr_char == extracted_p:
                    neighbours.append((target, 0, p))  # sync_e cost = 0
            if "del" in p and curr_char is not None:
                extracted_p = int(p.replace("del_", ""))
                if curr_char == extracted_p:
                    neighbours.append((target, 1, p))  # del_e cost = 1
            if "add" in p:
                extracted_p = int(p.replace("add_", ""))
                if f"sync_{extracted_p}" not in set(inputs) and f"add_{extracted_p}" not in set(inputs):
                    neighbours.append((target, 1, p))  # add_e cost = 1

        return neighbours

    def alignment_length(curr_alignment):
        return sum(1 if "sync" in c or "add" in c else -1 for c in curr_alignment)

    if origin_state not in dfa.states or target_state not in dfa.states:
        warnings.warn('Origin or target state not in automaton. Returning None.')
        return None

    visited = set()
    paths = []
    heap_counter = 0

    # Modify to add heuristic to the cost calculation
    heapq.heappush(paths, (0, heap_counter, origin_state, [origin_state], [], remaining_trace, visited))
    
    pbar_counter = 0
    while paths and pbar_counter < max_steps:
        pbar_counter += 1
        if pbar_counter % 1000 == 0:
            print(f"Steps: {pbar_counter}")
            print(f"Remaining paths: {len(paths)}")
            print(f"Remaining trace: {len(remaining_trace)}")

        cost, _, current_state, path, inputs, remaining_trace, visited = heapq.heappop(paths)

        curr_alignment_length = alignment_length(inputs)
        if current_state == target_state and not remaining_trace and curr_alignment_length >= min_alignment_length:
            return tuple(inputs)

        if remaining_trace:
            curr_char = remaining_trace[-1]
            neighbours = get_constrained_neighbours(current_state, curr_char)
        else:
            neighbours = get_constrained_neighbours(current_state, None)

        for neighbour, action_cost, action in neighbours:
            new_cost = cost + action_cost
            new_path = path + [neighbour]
            new_inputs = inputs + [action]

            new_visited = visited.copy()
            if (current_state.state_id, action) in new_visited:
                continue

            new_visited.add((current_state.state_id, action))
            new_remaining_trace = remaining_trace.copy()
            if "sync" in action or "del" in action:
                new_remaining_trace.pop()

            heap_counter += 1
            # Include heuristic cost in the priority queue
            priority = new_cost + heuristic(curr_inputs=inputs)
            heapq.heappush(paths, (priority, heap_counter, neighbour, new_path, new_inputs, new_remaining_trace, new_visited))

    warnings.warn(f'Maximum steps ({max_steps}) reached. Returning None.')
    return None

