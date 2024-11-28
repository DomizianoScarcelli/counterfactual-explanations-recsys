from copy import deepcopy
from typing import List, Tuple, Union

from aalpy.automata.Dfa import Dfa, DfaState
from torch import Tensor

from alignment.a_star import faster_a_star
from alignment.actions import Action, decode_action, print_action
from automata_learning.utils import invert_automata, run_automata
from exceptions import CounterfactualNotFound
from genetic.utils import Items, get_items
from type_hints import Trace, TraceSplit
from utils import printd


def augment_trace_automata(automata: Dfa, items: Items = Items.ML_1M) -> Dfa:
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
    alphabet = get_items(items)

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


def ingoing_states(dfa: Dfa, curr_state: DfaState):
    ingoing = set()
    for state in dfa.states:
        if curr_state in state.transitions.values():
            ingoing.add(state)
    return ingoing


def augment_constraint_automata(automata: Dfa, source_sequence: List[int]) -> Dfa:
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
    alphabet = automata.get_input_alphabet()
    trace_alphabet = set(source_sequence)

    # we can add anything from the alphabet
    add_propositions = {p: f"add_{p}" for p in alphabet}
    # We can only delete characters that are included in the sequence
    del_propositions = {p: f"del_{p}" for p in trace_alphabet}

    for state in automata.states:
        transitions_to_add = []
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


def constraint_aut_to_planning_aut(a_dfa: Dfa):
    """
    Given a constraint automaton `a_dfa` where character are of type `e`,
    `add_e` and `del_e`, it converts each `e` in `sync_e`.
    """
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
    for state in a_dfa.states:
        for p, target_state in state.transitions.copy().items():
            if "sync" in p:
                extracted_p = int(p.replace("sync_", ""))
                state.transitions[extracted_p] = target_state
                del state.transitions[p]


def compute_alignment_cost(alignment: Tuple[int]) -> int:
    """
    Computes the cost of the alignment:
        - ADD and DEL actions alone have cost 1.
        - A sequence of ADD-DEL or DEL-ADD has a combined cost of 1.
        - SYNC actions have cost 0.

    Args:
        alignment: A tuple of integers that represents Actions and can be decoded into action_type and number.

    Returns:
        The alignment cost.
    """
    cost = 0
    prev_action_type = None

    for encoded_e in alignment:
        action_type, _ = decode_action(encoded_e)
        if action_type in {Action.ADD, Action.DEL}:
            # If the current action pairs with the previous action (ADD-DEL or DEL-ADD), skip increment
            if prev_action_type in {Action.ADD, Action.DEL} and action_type != prev_action_type:
                prev_action_type = None  # Reset to avoid counting overlapping pairs
                continue
            cost += 1
        elif action_type == Action.SYNC:
            prev_action_type = None  # Reset since SYNC breaks the pairing
        else:
            prev_action_type = action_type  # Update previous action

        prev_action_type = action_type

    return cost


def trace_alignment(a_dfa_aug: Dfa, trace_split: Union[Trace, TraceSplit]):
    constraint_aut_to_planning_aut(a_dfa_aug)

    if not (isinstance(trace_split, tuple) and len(trace_split) == 3):
        trace_split = ([], trace_split, [])
    safe_trace_split: TraceSplit = tuple([int(c.item()) if isinstance(
        c, Tensor) else c for c in trace] for trace in trace_split)

    a_dfa_aug.reset_to_initial()

    min_length = len(safe_trace_split[0] + safe_trace_split[1])
    max_length = min_length

    alignment = faster_a_star(dfa=a_dfa_aug,
                              trace_split=safe_trace_split,
                              min_alignment_length=min_length,
                              max_alignment_length=max_length)
    if alignment is None:
        raise CounterfactualNotFound("No best path found")
    printd(f"Alignment is: {[print_action(a) for a in alignment]}")
    # print("Alignments is: ", [f"{act_str(decode_action(a)[0])}_{decode_action(a)[1]}" for a in alignment])
    planning_aut_to_constraint_aut(a_dfa_aug)
    aligned_trace = align(alignment)
    aligned_accepts = run_automata(a_dfa_aug, aligned_trace)
    # TODO: insert it back
    assert aligned_accepts, "Automa should accept aligned trace"
    cost = compute_alignment_cost(alignment)
    return aligned_trace, cost, alignment


def trace_disalignment(a_dfa_aug: Dfa, trace_split: Union[Trace, TraceSplit]):
    """
    It finds the changes with minimum cost to do to a trace accepted by the automata in order for it to not be
    accepted anymore. 
    """
    a_dfa_aug = deepcopy(a_dfa_aug)
    invert_automata(a_dfa_aug)
    aligned_trace, cost, alignment = trace_alignment(a_dfa_aug, trace_split)

    # TODO: maybe insert this back
    # dfa_rejects = not run_automata(a_dfa_aug, trace)
    # if not dfa_rejects:
    #     raise DfaNotRejecting("Dfa is not rejecting original sequence")
    # dfa_accepts = run_automata(a_dfa_aug, aligned_trace)
    # if not dfa_accepts:
    #     raise DfaNotAccepting("Dfa is not accepting counterfactual sequence")
    return aligned_trace, cost, alignment


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
    for encoded_action in alignment:
        action_type, e = decode_action(encoded_action)
        if action_type == Action.SYNC:
            aligned_trace.append(e)
        if action_type == Action.ADD:
            aligned_trace.append(e)
    return aligned_trace
