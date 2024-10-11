from aalpy.automata.Dfa import Dfa, DfaState
from dataset_generator import NumItems
from tqdm import tqdm
from typing import List, Tuple, Set, Dict, Optional
import warnings
from automata_learning import run_automata
from copy import deepcopy

DEBUG = True


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
    alphabet = [i for i in range(1, num_items.value)]
    
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
    del_propositions = {p: f"del_{p}" for p in alphabet}

    for state in automata.states:
        transitions_to_add = []
        #TODO: this shoube be trace_alphabet, I'm trying with alphabet, but I should try with universe
        for trace_p in alphabet:
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


def get_shortest_alignment_dijkstra(dfa, origin_state, target_state, remaining_trace: List[int], added_symbols: set):
    """
    Dijkstra's algorithm to find the shortest path (alignment) between two states in the DFA, 
    considering that sync_e actions have a cost of 0 and add_e or del_e actions have a cost of 1.
    """
    remaining_trace = remaining_trace.copy()

    # constraint to keep the aligned trace at a certain length
    # TODO: implement it in the code
    min_result_length = 50 

    def get_constrained_neighbours(state, curr_char: Optional[int]):
        neighbours = []
        for p, target in state.transitions.items():
            if p in added_symbols:
                continue

            if p in visited[state.state_id]:
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
                print(f"[DEBUG {p}] synced_added is: ", set(inputs))
                extracted_p = int(p.replace("add_", ""))
                if f"sync_{extracted_p}" not in set(inputs) and f"add_{extracted_p}" not in set(inputs):
                    neighbours.append((target, 1, p))  # add_e or del_e cost = 1

        printd(f"Neighbours for char: {curr_char} in state {state.state_id} are {[(s.state_id, c, p) for (s, c, p) in neighbours]}")
        return neighbours

    if origin_state not in dfa.states or target_state not in dfa.states:
        warnings.warn('Origin or target state not in automaton. Returning None.')
        return None

    # List of paths: each entry is (cumulative_cost, state, path_to_state, input_sequence, remaining_trace)
    # explored = set()
    # to avoid infinite loops, track visited edges from states
    visited: Dict[DfaState, Set[int]] = {state.state_id: set() for state in dfa.states}
    paths = [(0, origin_state, [origin_state], [], remaining_trace, visited)]


    while paths:
        # Find the path with the lowest cumulative cost
        paths.sort(key=lambda x: x[0])
        cost, current_state, path, inputs, remaining_trace, visited = paths.pop(0)

        # if current_state in explored:
        #     continue
        # explored.add(current_state)

        # If the current state is also the target state, I can exit only if I
        # don't have any characters to read in the sequence
        if current_state == target_state and len(remaining_trace) == 0:
            return tuple(inputs)

        # print(f"Remaining paths: {len(paths)}")
        # print(f"Current cost: {cost}")
        
        # if there are still characters to read, you can do all actions
        if remaining_trace:
            curr_char = remaining_trace[-1]
            # printd(f"""
            #       ----------
            #       [DEBUG] DIJKSTRA STATE
            #       Cost: {cost}
            #       Current Char: {curr_char}
            #       Current state:{current_state.state_id}
            #       Path:{[s.state_id for s in path]}
            #       Inputs:{inputs}
            #       Remaining Trace: {remaining_trace}
            #       """)
            neighbours = get_constrained_neighbours(current_state, curr_char)
        else:
            # otherwise you can only do add_e
            neighbours = get_constrained_neighbours(current_state, curr_char=None)

        for neighbour, action_cost, action in neighbours:
            new_cost = cost + action_cost
            new_path = path + [neighbour]
            new_inputs = inputs + [action]
            
            new_visited = deepcopy(visited)
            if action in new_visited[current_state.state_id]:
                printd(f"Action {action} already done in {current_state.state_id}")
                continue
            new_visited[current_state.state_id].add(action)

            new_remaining_trace = remaining_trace.copy()
            if "sync" in action or "del" in action:
                new_remaining_trace.pop()
            
            paths.append((new_cost, neighbour, new_path, new_inputs, new_remaining_trace, new_visited))



    return None

def negate_automata(automata: Dfa):
    """
    
    """
    pass

def constraint_aut_to_planning_aut(a_dfa: Dfa):
    print("Replacing e with sync_e...")
    for state in a_dfa.states:
        for p, target_state in state.transitions.copy().items():
            if type(p) is int:
                state.transitions[f"sync_{p}"] = target_state
                del state.transitions[p]

def planning_aut_to_constraint_aut(a_dfa: Dfa):
    print("Replacing sync_e with e...")
    for state in a_dfa.states:
        for p, target_state in state.transitions.copy().items():
            if "sync" in p:
                extracted_p = int(p.replace("sync_", ""))
                state.transitions[extracted_p] = target_state
                del state.transitions[p]

def compute_alignment_cost(alignment) -> int:
    cost = 0
    for s in alignment:
        if "add" in s or "del" in s:
            cost += 1
    return cost

def run_trace_alignment(a_dfa_aug: Dfa, trace: List[int]):
    constraint_aut_to_planning_aut(a_dfa_aug)
    a_dfa_aug.reset_to_initial()
    final_states = set(state for state in a_dfa_aug.states if state.is_accepting)
    curr_run = []
    added_symbols = set()
    running_trace = list(reversed(trace)).copy()
    #TODO:
    # This has to be solved in another way, since if I execute the whole
    # sequence than I cannot go back.
    # run_automata(a_dfa_aug, trace)
    print(f"Starting state: ", a_dfa_aug.initial_state.state_id)
    print(f"State after reading sequence: ", a_dfa_aug.current_state.state_id)
    while len(running_trace) > 0:
        accepting_runs = set()
        for f_state in final_states:
            shortest = get_shortest_alignment_dijkstra(a_dfa_aug, a_dfa_aug.current_state, f_state, running_trace, added_symbols)
            if shortest:
                accepting_runs.add(shortest)
            print(accepting_runs)
        # compute runs cost
        best_run = min([(run, compute_alignment_cost(run)) for run in accepting_runs], key=lambda x: x[1])
        #insert the best action into the current run
        best_action = best_run[0][0]
        # add del_e if best_action = add_e
        # TODO: this is a workaround for now
        # if "add" in best_action:
        #     del_action = f"del_{running_trace[-1]}"
        #     curr_run.append(del_action)
        #     added_symbols.add(del_action)
        #     a_dfa_aug.step(del_action)
        
        added_symbols.add(best_action)
        curr_run.append(best_action)
        # update running_trace
        a_dfa_aug.step(best_action)
        running_trace.pop()
    cost = compute_alignment_cost(curr_run)
    # assert a_dfa_aug.current_state.is_accepting, "DFA is not accepting the aligned trace"
    return curr_run, cost


def run_trace_disalignment(a_dfa_aug: Dfa, trace: List[int]):
    """
    It finds the changes with minimum cost to do to a trace accepted by the automata in order for it to not be
    accepted anymore. 
    """
    pass


def align(trace: List[int], alignment: tuple) -> List[int]:
    aligned_trace = []
    i = 0
    for j, a_char in enumerate(alignment):
        if "sync" in a_char:
            char = int(a_char.replace("sync_", ""))
            aligned_trace.append(char)
        if "add" in a_char:
            char = int(a_char.replace("add_", ""))
            aligned_trace.append(char)
        if "del" in a_char:
            pass
    return aligned_trace


