from aalpy.learning_algs import run_RPNI
from aalpy.automata.Dfa import Dfa, DfaState
from dataset_generator import NumItems
from recommenders.test import model_predict, load_dataset, load_data, generate_model
import pickle
import os
from typing import Union, List
from aalpy.utils.HelperFunctions import make_input_complete
from recbole.config import Config
from tqdm import tqdm
import random
from collections import deque

automata_save_path = "automata.pickle"

def generate_automata(dataset, load_if_exists: bool=True) -> Union[None, Dfa]:
    if os.path.exists(automata_save_path) and load_if_exists:
        print("Loaded existing automata")
        dfa = load_automata()
        return dfa
    print("Existing automata not found, generating a new one based on the provided dataset")
    dfa = run_RPNI(data=dataset, automaton_type="dfa")
    if dfa is None:
        return 
    save_automata(dfa)
    print(f"Automata saved at {automata_save_path}")
    return dfa

def save_automata(automata):
    with open(automata_save_path, "wb") as f:
        pickle.dump(automata, f)

def load_automata():
    with open(automata_save_path, "rb") as f:
        return pickle.load(f)


def generate_syntetic_point(min_value:int=1, max_value: int=NumItems.ML_1M.value, length: int = 50):
    point = []
    while len(point) < length:
        item = random.randint(min_value, max_value)
        if item not in point:
            point.append(item)
    return point
    

def run_automata(automata: Dfa, input: List[int]):
    automata.reset_to_initial()
    # automata.execute_sequence(origin_state=automata.current_state, seq=input)
    # return automata.current_state.is_accepting
    result = False
    for char in input:
        try:
            result = automata.step(char)
        except KeyError:
            #TODO: see how to handle this case
            continue
            # equivalent to go in sink state and early return
            print(f"Unknown character: {char}, rejecting.")
            return False
    return result


def generate_automata_from_dataset(dataset, load_if_exists: bool=True) -> Dfa:
    """
    Given a dataset with the following syntax:
        ([(torch.tensor([...]), good_label), ...],
          [(torch.tensor([...]), bad_label), ...])
    it learns a DFA that accepts good points and rejects bad points
    """
    good_points, bad_points = dataset
    data = [ (seq[0].tolist(), True) for seq in good_points ] + [ (seq[0].tolist(), False) for seq in bad_points ]
    dfa = generate_automata(data, load_if_exists)
    if dfa is None:
        raise RuntimeError("DFA is None, aborting")
    dfa = make_input_complete(dfa)
    assert dfa.is_input_complete(), "Dfa is not input complete"
    return dfa


def generate_single_accepting_sequence_dfa(sequence):
    """
    Generates a DFA that only accepts the input sequence.
    """
    # Create the initial state
    initial_state = DfaState('q0')
    
    # Create the states for each step in the sequence
    current_state = initial_state
    states = [initial_state]
    
    # For each character in the sequence, create a state and add transitions
    for i, symbol in enumerate(sequence):
        next_state = DfaState(f'q{i + 1}')
        current_state.transitions[symbol] = next_state
        states.append(next_state)
        current_state = next_state
    
    # Final state is the accepting state
    accepting_state = current_state
    accepting_state.is_accepting = True
    
    # Create a reject state for invalid transitions
    reject_state = DfaState('reject')
    states.append(reject_state)
    
    # Set up transitions to the reject state for all incorrect inputs

    # assuming sequence contains all possible symbols, since symbols not in the
    # sequence will be ignored and will lead to a rejecting state
    all_symbols = set(sequence)
    for state in states:
        for symbol in all_symbols:
            if symbol not in state.transitions:
                state.transitions[symbol] = reject_state
        reject_state.transitions = {symbol: reject_state for symbol in all_symbols}

    # Return the DFA
    dfa = Dfa(initial_state, states)
    return dfa


if __name__ == "__main__":
    print(f"Generating automata from saved dataset")

    #Remove non-determinism
    g,b= load_dataset(load_path="saved/counterfactual_dataset.pickle")
    new_g, new_b = [], []
    ids = set()
    for p, l in g:
        if tuple(p.tolist()) in ids:
            continue
        new_g.append((p,l))
        ids.add(tuple(p.tolist()))
    for p, l in b:
        if tuple(p.tolist()) in ids:
            continue
        new_b.append((p,l))
        ids.add(tuple(p.tolist()))

    dataset = (new_g, new_b)
    dfa = generate_automata_from_dataset(dataset, load_if_exists=False)
    # dfa.visualize()


