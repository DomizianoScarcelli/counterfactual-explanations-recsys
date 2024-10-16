import os
import pickle
import random
from typing import List, Tuple, Union

from aalpy.automata.Dfa import Dfa, DfaState
from aalpy.learning_algs import run_RPNI
from aalpy.utils.HelperFunctions import make_input_complete

from dataset_generator import NumItems
from recommenders.generate_dataset import load_dataset
from trace_alignment import augment_constraint_automata
from type_hints import Dataset


def generate_automata(dataset, load_if_exists: bool=True, save_path: str="automata.pickle") -> Union[None, Dfa]:
    if os.path.exists(os.path.join("saved_automatas", save_path)) and load_if_exists:
        print("Loaded existing automata")
        dfa = load_automata(save_path)
        return dfa
    print("Existing automata not found, generating a new one based on the provided dataset")
    dfa = run_RPNI(data=dataset, automaton_type="dfa")
    if dfa is None:
        return 
    save_automata(dfa, save_path)
    print(f"Automata saved at {save_path}")
    return dfa

def save_automata(automata, save_path):
    with open(os.path.join("saved_automatas", save_path), "wb") as f:
        pickle.dump(automata, f)

def load_automata(load_path):
    with open(os.path.join("saved_automatas", load_path), "rb") as f:
        return pickle.load(f)

def generate_automata_from_dataset(dataset, load_if_exists: bool=True, save_path: str="automata.pickle") -> Dfa:
    """
    Given a dataset with the following syntax:
        ([(torch.tensor([...]), good_label), ...],
          [(torch.tensor([...]), bad_label), ...])
    it learns a DFA that accepts good points and rejects bad points
    """
    good_points, bad_points = dataset
    data = [ (seq[0].tolist(), True) for seq in good_points ] + [ (seq[0].tolist(), False) for seq in bad_points ]
    dfa = generate_automata(data, load_if_exists, save_path)
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


def learning_pipeline(source: List[int], dataset: Tuple[Dataset, Dataset]) -> Dfa:
    t_dfa = generate_single_accepting_sequence_dfa(source)
    a_dfa = generate_automata_from_dataset(dataset, load_if_exists=False)
    a_dfa_aug = augment_constraint_automata(a_dfa, t_dfa)
    return a_dfa_aug


if __name__ == "__main__":
    print(f"Generating automata from saved dataset")

    #Remove non-determinism
    dataset = load_dataset(load_path="saved/counterfactual_dataset.pickle")
    # dataset = make_deterministic(dataset)

    dfa = generate_automata_from_dataset(dataset, load_if_exists=False)
    # dfa.visualize()


