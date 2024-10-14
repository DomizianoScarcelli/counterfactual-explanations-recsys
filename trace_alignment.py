from aalpy.automata.Dfa import Dfa, DfaState
from automata_utils import invert_automata
from dataset_generator import NumItems
from tqdm import tqdm
from typing import List, Tuple
from automata_utils import run_automata
from graph_search import (decode_action, get_shortest_alignment_dijkstra, 
                          get_shortest_alignment_a_star)
from copy import deepcopy
from graph_search import Action, decode_action, act_str
from constants import MAX_LENGTH

DEBUG = False


def printd(statement):
    if DEBUG:
        print(statement)

def augment_trace_automata(automata: Dfa, num_items: NumItems=NumItems.ML_1M) -> Dfa:
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
    alphabet = [i for i in range(0, num_items.value)] #TODO: when zero trimming will work, put a 1 here
    
    # Create the new repair propositions
    add_propositions = {p: f"add_{p}" for p in alphabet}
    del_propositions = {p: f"del_{p}" for p in alphabet}

    # Step 2: Augment the automaton by adding new transitions for `del_p` and `add_p`
    for state in automata.states:
        # Step 2.1: For every existing transition (q, p, q'), add (q, del_p, q')
        transitions_to_add = []
        for p, target_state in state.transitions.items():
            del_p = del_propositions[p]
            # Add (q, del_p, q') transition for each (q, p, q')
            transitions_to_add.append((del_p, target_state))

        for p in alphabet:
            if p not in state.transitions:
                add_p = add_propositions[p]
                # Create a self-loop for add_p 
                transitions_to_add.append((add_p, state))

        # Now add the new transitions to the state
        for new_transition_symbol, target_state in transitions_to_add:
            state.transitions[new_transition_symbol] = target_state

    return automata



def augment_constraint_automata(automata: Dfa, trace_automaton: Dfa) -> Dfa:
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
    # alphabet = [i for i in range(1, NumItems.ML_1M.value)]
    alphabet = automata.get_input_alphabet()
    trace_alphabet = trace_automaton.get_input_alphabet()
    
    # Create the new repair propositions
    add_propositions = {p: f"add_{p}" for p in alphabet}
    del_propositions = {p: f"del_{p}" for p in trace_alphabet}

    for state in automata.states:
        transitions_to_add = []
        #TODO: I should try with universe
        for trace_p in trace_alphabet:
            del_p = del_propositions[trace_p]
            transitions_to_add.append((del_p, state))

        for p, target_state in state.transitions.items():
            add_p = add_propositions[p]
            # Add (q, add_p, q') transition for each (q, p, q') in transitions
            transitions_to_add.append((add_p, target_state))
    
        # Now add the new transitions to the state
        for new_transition_symbol, target_state in transitions_to_add:
            state.transitions[new_transition_symbol] = target_state

    return automata

def _deprecated_create_intersection_automata(a_aug: Dfa, t_aug: Dfa) -> Dfa:
    """
    Given two automatas a_aug and t_aug, it returns the intersection of those automatas
    """
    states = set()
    state_map = {}
    for a_state in a_aug.states:
        for t_state in t_aug.states:
            is_accepting = a_state.is_accepting and t_state.is_accepting
            new_state = DfaState((a_state, t_state), is_accepting=is_accepting)
            state_map[(a_state, t_state)] = new_state
            states.add(new_state)
    assert len(states) == len(a_aug.states) * len(t_aug.states)
    accepting_states = set({s for s in states if s.is_accepting})
    printd(f"Automa has {len(accepting_states)} accepting states")
    
    a_alph = set(a_aug.get_input_alphabet()) 
    t_alph = set(t_aug.get_input_alphabet())
    alphabet = a_alph | t_alph
    
    added = 0
    for state in tqdm(states, desc="Creating states..."): 
        a_state, t_state = state.state_id
        for symbol in alphabet:
            a_target_state = a_state.transitions.get(symbol, None)
            t_target_state = t_state.transitions.get(symbol, None)
            if a_target_state and t_target_state:
                is_accpeting = a_target_state.is_accepting and t_target_state.is_accepting
                state.transitions[symbol] = state_map[(a_target_state, t_target_state)]
                added += 1

    printd(f"Added {added} transitions!")
    initial_state = DfaState(a_aug.initial_state, t_aug.initial_state)
    dfa = Dfa(initial_state=initial_state, states=states)
    return dfa


def create_intersection_automata(dfa1: Dfa, dfa2: Dfa) -> Dfa:
    """
    Just another way of doing `_deprecated_create_intersection_automata`
    """
    state_map = {}
    new_states = set()

    # Create initial state in the intersection DFA
    initial_state = DfaState((dfa1.initial_state, dfa2.initial_state),
                             is_accepting=(dfa1.initial_state.is_accepting and dfa2.initial_state.is_accepting))
    state_map[(dfa1.initial_state, dfa2.initial_state)] = initial_state
    new_states.add(initial_state)

    # Queue for breadth-first search (BFS) over state pairs
    queue = [(dfa1.initial_state, dfa2.initial_state)]
    alphabet = set(dfa1.get_input_alphabet()) & set(dfa2.get_input_alphabet())
    while queue:
        (state1, state2) = queue.pop(0)
        current_state = state_map[(state1, state2)]

        # Get common input alphabet for both DFAs
        for symbol in alphabet:
            # Get transitions for the current symbol
            target1 = state1.transitions.get(symbol, None)
            target2 = state2.transitions.get(symbol, None)

            # If both DFAs have transitions for the symbol
            if target1 and target2:
                if (target1, target2) not in state_map:
                    # Create the new state in the intersection DFA
                    is_accepting = target1.is_accepting and target2.is_accepting
                    new_state = DfaState((target1, target2), is_accepting=is_accepting)
                    state_map[(target1, target2)] = new_state
                    new_states.add(new_state)
                    queue.append((target1, target2))

                # Add the transition to the current state
                current_state.transitions[symbol] = state_map[(target1, target2)]

    # Create and return the intersection DFA
    dfa = Dfa(initial_state=initial_state, states=list(new_states))
    printd(f"Intersection DFA automata alphabet is: {dfa.get_input_alphabet()}")
    return dfa

def constraint_aut_to_planning_aut(a_dfa: Dfa):
    """
    Given a constraint automaton `a_dfa` where character are of type `e`,
    `add_e` and `del_e`, it converts each `e` in `sync_e`.
    """
    print("Replacing e with sync_e...")
    for state in a_dfa.states:
        for p, target_state in state.transitions.copy().items():
            if type(p) is int:
                state.transitions[f"sync_{p}"] = target_state
                del state.transitions[p]

def planning_aut_to_constraint_aut(a_dfa: Dfa):
    """
    Given a constraint automaton `a_dfa` where character are of type `sync_e`,
    `add_e` and `del_e`, it converts each `sync_e` in `e`.
    """
    print("Replacing sync_e with e...")
    for state in a_dfa.states:
        for p, target_state in state.transitions.copy().items():
            if "sync" in p:
                extracted_p = int(p.replace("sync_", ""))
                state.transitions[extracted_p] = target_state
                del state.transitions[p]

def compute_alignment_cost(alignment: Tuple[int]) -> int:
    """
    Computes the cost of the alignment:
        - add_e and del_e actions have cost 1
        - sync_e actions have cost 0

    :param alignment Tuple[int]: a tuple of integers that represents Actions and can be decoded in action_type and number.
    :rtype int: the alignment cost.
    """
    cost = 0
    for encoded_e in alignment:
        action_type, _ = decode_action(encoded_e) 
        if action_type in {Action.ADD, Action.DEL}:
            cost += 1
    return cost

def trace_alignment(a_dfa_aug: Dfa, trace: List[int]):
    """
    """
    # min_length = len(trace)
    min_length = len(trace)
    max_length = MAX_LENGTH
    print(f"Expected length interval: ({min_length}, {max_length})")
    constraint_aut_to_planning_aut(a_dfa_aug)
    remaining_trace = list(trace)
    final_states = set(s for s in a_dfa_aug.states if s.is_accepting)
    print(f"Final states are {[s.state_id for s in final_states]}")
    a_dfa_aug.reset_to_initial()
    alignment = get_shortest_alignment_dijkstra(dfa=a_dfa_aug, 
                                              origin_state=a_dfa_aug.initial_state, 
                                              target_states=final_states,
                                              remaining_trace=remaining_trace,
                                              min_alignment_length=min_length,
                                              max_alignment_length=max_length)
    assert alignment is not None, "No best path found"
    print("Alignments is: ", [f"{act_str(decode_action(a)[0])}_{decode_action(a)[1]}" for a in alignment])
    planning_aut_to_constraint_aut(a_dfa_aug)
    aligned_trace = align(alignment)
    aligned_accepts = run_automata(a_dfa_aug, aligned_trace)
    assert aligned_accepts, "Automa should accept aligned trace"
    cost = compute_alignment_cost(alignment)
    print("Alignment cost: ", cost)
    # aligned_traces.append((aligned_trace, cost))
    # best_alignment, best_cost = min(aligned_traces, key=lambda x: x[1])
    return aligned_trace, cost


def trace_disalignment(a_dfa_aug: Dfa, trace: List[int]):
    """
    It finds the changes with minimum cost to do to a trace accepted by the automata in order for it to not be
    accepted anymore. 
    """
    a_dfa_aug = deepcopy(a_dfa_aug)
    invert_automata(a_dfa_aug)
    return trace_alignment(a_dfa_aug, trace)


def align(alignment: Tuple[int]) -> List[int]:
    """
    Converts an alignment into an aligned trace

    Args:
        alignment: A tuple of integers that represent Actions and that can be
        decoded into action_type, action_number

    Returns:
        The aligned trace
    """
    aligned_trace = []
    print(f"[align] Alignment is: {alignment}")
    for encoded_action in alignment:
        action_type, e = decode_action(encoded_action)
        if action_type == Action.SYNC:
            aligned_trace.append(e)
        if action_type == Action.ADD:
            aligned_trace.append(e)
    return aligned_trace


