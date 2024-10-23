from collections import deque
from typing import List, Optional, Set

from aalpy.automata.Dfa import DfaState

from alignment.actions import Action, decode_action


def hops(curr_state, remaining_trace, target_states):
    #NOTE: this is the best for now
    def _bfs(state: DfaState, trace: Optional[List[int]], target_states: Set[DfaState]):
        # Execute the most amount of sync actions on the trace
        if trace:
            for c in trace:
                sync_c = f"sync_{c}"
                if sync_c in state.transitions:
                    state = state.transitions[sync_c]
                else:
                    break

        # Perform BFS from the current state to any target state
        queue = deque([(state, 0)])  # Queue of (current_state, hop_length)
        visited = set()  # Set of visited states
        visited.add(state.state_id)

        while queue:
            current_state, hop_length = queue.popleft()

            # Check if the current state is one of the target states
            if current_state in target_states:
                return hop_length

            # Process transitions for sync, add, and del actions
            for next_state in current_state.transitions.values():
                if next_state.state_id not in visited:
                    visited.add(next_state.state_id)
                    queue.append((next_state, hop_length + 1))

        return float("inf")  # No target state is reachable

    return _bfs(curr_state, remaining_trace, target_states)
    
def add_dels(inputs):
    # NOTE: this is worse than no heuristic
    adds, dels, syncs = 0, 0, 0
    cost = 0
    for i in inputs:
        action_type, _ = decode_action(i)
        if action_type == Action.ADD:
            adds += 1
        elif action_type == Action.SYNC:
            syncs += 1
        elif action_type == Action.DEL:
            dels += 1
    if (adds + dels) % 2 != 0:
        cost += adds + dels
    return cost

def none():
    return 0
