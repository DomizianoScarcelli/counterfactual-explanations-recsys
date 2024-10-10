from aalpy.automata.Dfa import Dfa, DfaState
from dataset_generator import NumItems
import itertools
from tqdm import tqdm
from typing import List
import warnings


def augment_trace_automata(automata: Dfa) -> Dfa:
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
        for trace_p in trace_alphabet:
            del_p = del_propositions[trace_p]
            transitions_to_add.append((del_p, state))

        for p, target_state in state.transitions.items():
            add_p = add_propositions[p]
            # Add (q, add_p, q') transition for each (q, p, q') in transitions
            #TODO: don't know if this is correct
            transitions_to_add.append((add_p, target_state))
    
        # Now add the new transitions to the state
        for new_transition_symbol, target_state in transitions_to_add:
            state.transitions[new_transition_symbol] = target_state

    return automata

def _deprecated_create_planning_automata(a_aug: Dfa, t_aug: Dfa) -> Dfa:
    #TODO: fix it, since it should accept if both the automatas accpet, but for
    # not this is not the case
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
    print(f"Automa has {len(accepting_states)} accepting states")
    
    a_alph = set(a_aug.get_input_alphabet()) 
    t_alph = set(t_aug.get_input_alphabet())
    alphabet = a_alph & t_alph
    
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

    print(f"Added {added} transitions!")
    initial_state = DfaState(a_aug.initial_state, t_aug.initial_state)
    dfa = Dfa(initial_state=initial_state, states=states)
    return dfa


def create_intersection_automata(dfa1: Dfa, dfa2: Dfa) -> Dfa:
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
    return Dfa(initial_state=initial_state, states=list(new_states))

# def make_planning_automata_correct(p_dfa: Dfa):
#     print("Correcting p_dfa...")
#     for state in p_dfa.states:
#         for p, target_state in state.transitions.copy().items():
#             if type(p) is int:
#                 state.transitions[f"sync_{p}"] = target_state
#                 state.transitions[f"add_{p}"] = target_state
#                 del state.transitions[p]

def get_shortes_alignment(dfa: Dfa, origin_state: DfaState, target_state: DfaState, remaining_trace: List[int]):
    """
    Breath First Search over the automaton to find the optimal alignment. 
    """
    #TODO: insert constraints in order to do a sync_s only if the remaining
    # trace has that character in that place, otherwise only del_s or add_s

    def get_constrained_neighbours(state: DfaState):
        neighbours = set()
        debug_mapping = {}
        curr_char = remaining_trace[-1]
        for p, target_state in state.transitions.items():
            if type(p) is int and curr_char == p:
                neighbours.add(target_state)
                debug_mapping[p] = target_state
            if type(p) is not int:
                neighbours.add(target_state)
                debug_mapping[p] = target_state
        # print(f"Current char = {curr_char}, neighbours: {list(debug_mapping.keys())}")
        return neighbours

    if origin_state not in dfa.states or target_state not in dfa.states:
        warnings.warn('Origin or target state not in automaton. Returning None.')
        return None

    explored = []
    queue = [[origin_state]]

    if origin_state == target_state:
        return ()

    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            neighbours = get_constrained_neighbours(node)
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == target_state:
                    acc_seq = new_path[:-1]
                    inputs = []
                    for ind, state in enumerate(acc_seq):
                        inputs.append(next(key for key, value in state.transitions.items()
                                           if value == new_path[ind + 1]))
                    return tuple(inputs)

            # mark node as explored
            explored.append(node)
    return None

def run_trace_alignment(p_dfa: Dfa, trace: List[int]):
    # TODO: remember that the goal is to "disalign" a trace: from being good and accepted to being rejected, 
    # with minimum cost
    p_dfa.reset_to_initial()
    final_states = set(state for state in p_dfa.states if state.is_accepting)
    accepting_runs = set()
    print(f"Final states are: {[(s.state_id[0].state_id, s.state_id[1].state_id) for s in final_states]}")
    running_trace = list(reversed(trace)).copy()
    for s in trace:
        try:
            p_dfa.step(s)
            running_trace.pop()
        except KeyError:
            pass
        saved_state = p_dfa.current_state
        for f_state in final_states:
            shortest = p_dfa.get_shortest_path(saved_state, f_state)
            shortest = get_shortes_alignment(p_dfa, saved_state, f_state, trace)
            accepting_runs.add(shortest)
    return [run for run in accepting_runs if run is not None]



