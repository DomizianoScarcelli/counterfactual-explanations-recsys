import heapq
from aalpy.automata.Dfa import Dfa, DfaState
from typing import List, Optional, Set, Callable
import warnings
# from memory_profiler import memory_usage
# import tracemalloc

# Store actions as raw bits for memory efficiency
class Action:
    SYNC = 0b00  # 0
    DEL = 0b01   # 1
    ADD = 0b10   # 2

def encode_action(action_type: int, number: int) -> int:
    return (action_type << 13) | number  # 2 bits for action type, 13 bits for number

def encode_action_str(action: str) -> int:
    action_type = None
    number = None
    if "sync" in action:
        action_type = Action.SYNC
        number = int(action.replace("sync_", ""))
    elif "del" in action:
        action_type = Action.DEL
        number = int(action.replace("del_", ""))
    elif "add" in action:
        action_type = Action.ADD
        number = int(action.replace("add_", ""))
    
    assert action_type is not None, f"Action '{action}' not supported"
    assert number is not None, f"Number not extracted from action '{action}'"
    
    return encode_action(action_type, number)  # Use the encode_action function

def decode_action(encoded_action: int):
    action_type = (encoded_action >> 13) & 0b11  # Extract the action type (2 bits)
    number = encoded_action & 0x1FFF  # Extract the number (13 bits)
    return action_type, number

def act_str(action: int):
    if action == Action.SYNC:
        return "sync"
    if action == Action.ADD:
        return "add"
    if action == Action.DEL:
        return "del"

# def log_memory_usage(message: str):
#     mem = memory_usage(-1, interval=0.1, timeout=1)
#     print(f"{message}: Current memory usage: {mem[0]:.4f} MB")

# def analyze_tracemalloc():
#     snapshot = tracemalloc.take_snapshot()
#     top_stats = snapshot.statistics('lineno')
#     print("[Top 10 Memory Allocations]")
#     for stat in top_stats[:10]:
#         print(stat)

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
    # tracemalloc.start()
    
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
            
            encoded_p = encode_action_str(p) if isinstance(p, str) else p
            action_type, e = decode_action(encoded_p)
            
            if action_type == Action.SYNC and curr_char is not None:
                add_e = encode_action(Action.ADD, e)
                already_added = p in inputs or add_e in inputs
                if curr_char == e and not already_added:
                    neighbours.append((target, 0, encoded_p))  # sync_e cost = 0
            elif action_type == Action.DEL and curr_char is not None:
                if curr_char == e:
                    neighbours.append((target, 1, encoded_p))  # del_e cost = 1
            elif action_type == Action.ADD:
                sync_e = encode_action(Action.SYNC, e)
                already_added = p in inputs or sync_e in inputs
                if not already_added:
                    neighbours.append((target, 1, encoded_p))  # add_e cost = 1

        return neighbours

    def alignment_length(curr_alignment):
        return sum(1 for encoded_action in curr_alignment if decode_action(encoded_action)[0] in {Action.SYNC, Action.ADD})
    
    invalid_states = [s for s in target_states if s not in dfa.states]
    if origin_state not in dfa.states or invalid_states:
        warnings.warn('Origin or target state not in automaton. Returning None.')
        return None

    visited: Set[int] = set()
    paths = []
    heap_counter = 0

    heapq.heappush(paths, (0, heap_counter, origin_state, (origin_state,), (), remaining_trace_idx, visited))
    
    pbar_counter = 0
    while paths:
        pbar_counter += 1
        # if pbar_counter % 1000 == 0:
        #     print(f"Steps: {pbar_counter}")
        #     print(f"Remaining paths: {len(paths)}")
        #     print(f"Remaining trace idx: {remaining_trace_idx}")
            # print(f"Remaining trace: {remaining_trace[-remaining_trace_idx:]}")
            # log_memory_usage("")
            # analyze_tracemalloc()

        cost, _, current_state, path, inputs, remaining_trace_idx, visited = heapq.heappop(paths)

        curr_alignment_length = alignment_length(inputs)
        if current_state in target_states and remaining_trace_idx == 0:
            if (min_alignment_length is None or curr_alignment_length >= min_alignment_length) and \
               (max_alignment_length is None or curr_alignment_length <= max_alignment_length):
                return tuple(inputs)
        
        curr_char = remaining_trace[-remaining_trace_idx] if remaining_trace_idx > 0 else None
        neighbours = get_constrained_neighbours(current_state, curr_char)

        for neighbour, action_cost, action in neighbours:
            new_cost = cost + action_cost
            new_path = path + (neighbour,)  # This creates a new tuple (efficient due to reference copying)
            new_inputs = inputs + (action,)  # This creates a new tuple (efficient due to reference copying)

            new_visited = visited
            if (current_state.state_id, action) in new_visited:
                continue
            
            new_visited.add((current_state.state_id, action))

            new_remaining_trace_idx = remaining_trace_idx
            action_type, _ = decode_action(action)
            if action_type in {Action.SYNC, Action.DEL}:
                new_remaining_trace_idx -= 1

            heap_counter += 1
            priority = new_cost + heuristic(curr_inputs=new_inputs)
            heapq.heappush(paths, (priority, heap_counter, neighbour, new_path, new_inputs, new_remaining_trace_idx, new_visited))

    return None
