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

def augment_trace_automata(automata: Dfa):
    """
    Given an DFA `T` which only accepts a certain sequence `s`, it augments it
    according to the rules explained in the paper "On the Disruptive
    Effectiveness of Automated Planning for LTLf-Based Trace Alignment" by De
    Giacomo et al:
        1. Two new propositions `add_p` and `del_p` are added for each proposition p in
        the T alphabet;
        2. Two new transitions `(q, add_p, q)` and `(q, del_p, q')` are added for each
        proposition `p` in the `T` alphabet and for each state q in the transition
        function, if there is no transition `(q, p, q')` for all `q'` in the transition
        function.
    
    `add_p` and `del_p` are called "repair propositions", and their purpose it to
    mark the repairs done to the sequence.

    The augmented automaton `T+` will accept all sequences that are generated
    from the original sequence by marking addition and deletion of characters
    with the corresponding repair propositions.
    """
    # Alphabet is the universe of all the items
    alphabet = [i for i in range(1, NumItems.ML_1M.value)]
    
    # Create the new repair propositions
    add_propositions = {p: f"add_{p}" for p in alphabet}
    del_propositions = {p: f"del_{p}" for p in alphabet}

    # Step 2: Augment the automaton by adding new transitions for `del_p` and `add_p`
    for state in automata.states:
        # Step 2.1: For every existing transition (q, p, q'), add (q, del_p, q')
        transitions_to_add = []
        for p, target_state in state.transitions.items():
            del_p = del_propositions[p]
            # Add (q, del_p, q') transition, where q' is the same as for p
            transitions_to_add.append((del_p, target_state))

        # Step 2.2: For every symbol p in the alphabet, if no transition on p exists from q,
        # add (q, add_p, q) transition (a self-loop for simplicity)
        for p in alphabet:
            if p not in state.transitions:
                add_p = add_propositions[p]
                # Create a self-loop for add_p, or you can choose a different target state
                transitions_to_add.append((add_p, state))

        # Now add the new transitions to the state
        for new_transition_symbol, target_state in transitions_to_add:
            state.transitions[new_transition_symbol] = target_state

    # alphabet = set()  # Collect the alphabet of the DFA
    # for state in automata.states:
    #     alphabet.update(state.transitions.keys())
    # print(f"Augmented alphabet is: {alphabet}")
    return automata



def augment_constraint_automata(automata: Dfa):
    """
    Given an DFA `A` which only accepts good sequences, defined as those
    sequences which label is the same as the ground truth sequence it augments
    it according to the rules explained in the paper "On the Disruptive
    Effectiveness of Automated Planning for LTLf-Based Trace Alignment" by De
    Giacomo et al:
            1. Two new propositions `add_p` and `del_p` are added for each proposition p in
            the T alphabet;
            2. A new transitions `(q, del_p, q')` is added for each
            proposition `p` in the `T` alphabet and for each state q in the transition
            function; and a new transition `(q, add_p, q)` is added for all
            transitions (q, phi, q') such that p satisfies the formula phi.
    
    The `phi` formula is the constraint that checks a trace against `A`. It
    returns true only on good sequences.

    The augmented automaton `A+` will accept all sequences that satisfy `phi`
    and that have been obtained by repairing the original sequence `s`
    """
    # Alphabet is the universe of all the items
    alphabet = [i for i in range(1, NumItems.ML_1M.value)]
    
    # Create the new repair propositions
    add_propositions = {p: f"add_{p}" for p in alphabet}
    del_propositions = {p: f"del_{p}" for p in alphabet}

    # Step 2: Augment the automaton by adding new transitions for `del_p` and `add_p`
    for state in automata.states:
        # Step 2.1: For every existing transition (q, p, q'), add (q, del_p, q')
        transitions_to_add = []
        for p, target_state in state.transitions.items():
            del_p = del_propositions[p]
            # Add (q, del_p, q') transition, where q' is the same as for p
            transitions_to_add.append((del_p, target_state))

        # Step 2.2: For every symbol p in the alphabet, if the transition (q, add_p, q') 
        # creates a sequence that is still accepted by the formula, then add that transition.
        #TODO: this has to be implemented
        for p in alphabet:
            pass
    
        # Now add the new transitions to the state
        for new_transition_symbol, target_state in transitions_to_add:
            state.transitions[new_transition_symbol] = target_state

    # alphabet = set()  # Collect the alphabet of the DFA
    # for state in automata.states:
    #     alphabet.update(state.transitions.keys())
    # print(f"Augmented alphabet is: {alphabet}")
    return automata

def synchronous_product(dfa_A: Dfa, dfa_T: Dfa):
    #TODO: THIS HAS TO BE TESTED!!!
    """
    Compute the synchronous product of two DFAs A and T.
    The resulting DFA will only accept a string if both A and T accept it.
    """
    # Create a new DFA state for each pair of states from A and T
    new_states = {}
    
    def get_new_state(state_A, state_T):
        if (state_A, state_T) not in new_states:
            new_state = DfaState(f"({state_A.state_id}, {state_T.state_id})")
            new_state.is_accepting = state_A.is_accepting and state_T.is_accepting
            new_states[(state_A, state_T)] = new_state
        return new_states[(state_A, state_T)]
    
    # Initialize the product DFA with the pair of initial states
    initial_state = get_new_state(dfa_A.initial_state, dfa_T.initial_state)
    
    # Create a list of states to process
    to_process = [(dfa_A.initial_state, dfa_T.initial_state)]
    
    # While there are unprocessed states
    while to_process:
        state_A, state_T = to_process.pop()
        new_state = get_new_state(state_A, state_T)
        
        # For each symbol in the alphabet, create the transitions
        for symbol in state_A.transitions:
            if symbol in state_T.transitions:
                next_state_A = state_A.transitions[symbol]
                next_state_T = state_T.transitions[symbol]
                
                # Get or create the next state in the product DFA
                next_new_state = get_new_state(next_state_A, next_state_T)
                new_state.transitions[symbol] = next_new_state
                
                # If the new state hasn't been processed yet, add it to the list
                if (next_state_A, next_state_T) not in new_states:
                    to_process.append((next_state_A, next_state_T))
    
    # Return the product DFA
    return Dfa(initial_state, list(new_states.values()))

