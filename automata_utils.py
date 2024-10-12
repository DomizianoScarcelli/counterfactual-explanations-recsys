from typing import List
from collections import deque
from aalpy.automata.Dfa import Dfa

def has_path_to_accepting_state(automaton: Dfa, seq: List[int]):
    """
    Check if the automata, in the state after executing the sequence `seq`, is
    in a state which can reach an accepting state in some way
    """
    automaton.reset_to_initial()
    automaton.execute_sequence(origin_state=automaton.current_state, seq=seq)
    queue = deque([automaton.current_state])
    visited = set()
    alphabet = automaton.get_input_alphabet()
    while queue:
        current_state = queue.popleft()
        if current_state.is_accepting:
            return True
        visited.add(current_state)
        for symbol in alphabet:
            next_state = current_state.transitions.get(symbol)
            if next_state and next_state not in visited:
                queue.append(next_state)
    return False


def invert_automata(automata: Dfa):
    """
    Inverts the automata by inverting the accepting/rejecting states. Each word
    accepted by the automata will be rejected by the inverse automata and vice
    versa
    
    """
    pass
