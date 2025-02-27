import os
import pickle
from collections import deque
from pathlib import Path
from typing import List

from aalpy.automata.Dfa import Dfa
from torch import Tensor

from utils import printd


def save_automata(automata, save_path):
    with open(Path(f"saved_automatas/{save_path}"), "wb") as f:
        pickle.dump(automata, f)
        printd(f"Automata saved at {save_path}")


def load_automata(load_path):
    with open(Path(f"saved_automatas/{load_path}"), "rb") as f:
        return pickle.load(f)


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
    for state in automata.states:
        state.is_accepting = not state.is_accepting


def run_automata(automata: Dfa, input: list, final_reset: bool = True):
    automata.reset_to_initial()
    # automata.execute_sequence(origin_state=automata.current_state, seq=input)
    # return automata.current_state.is_accepting
    result = False
    if isinstance(input, Tensor):
        input = input.tolist()
    for char in input:
        result = automata.step(char)
    if final_reset:
        automata.reset_to_initial()
    return result
