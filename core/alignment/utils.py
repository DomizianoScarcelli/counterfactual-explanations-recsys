import heapq
from statistics import mean
from typing import List

import torch
from aalpy.automata.Dfa import DfaState

from config.constants import MAX_LENGTH
from core.alignment.actions import Action, decode_action
from core.models.utils import pad, trim


def postprocess_alignment(aligned: List[int]):
    if len(aligned) == MAX_LENGTH:
        return torch.tensor(aligned).unsqueeze(0).to(torch.int64)
    if len(aligned) < MAX_LENGTH:
        return pad(trim(torch.tensor(aligned)), MAX_LENGTH).unsqueeze(0).to(torch.int64)
    raise ValueError(f"Aligned length > {MAX_LENGTH}: {len(aligned)}")

def alignment_length(curr_alignment):
    return sum(1 for encoded_action in curr_alignment if decode_action(encoded_action)[0] in {Action.SYNC, Action.ADD})

def get_path_statistics(paths, max_alignment_length=None, min_alignment_length=None):
    """
    Function to get statistics about the paths in the priority queue.
    
    Parameters:
    paths -- the priority queue (list of tuples containing (cost, heap_counter, state, path, inputs, remaining_trace_idx, visited))
    max_alignment_length -- the maximum allowed alignment length
    min_alignment_length -- the minimum allowed alignment length
    
    Returns:
    Dictionary containing statistics about the paths.
    """
    if not paths:
        return {
            'num_paths': 0,
            'min_cost': None,
            'max_cost': None,
            'avg_cost': None,
            'mean_alignment_lengths': [],
            'num_paths_near_max_length': 0,
            'num_paths_near_min_length': 0
        }

    costs = [path[0] for path in paths]  # Extract the cost from each path
    alignment_lengths = [alignment_length(path[4]) for path in paths]  # Extract the alignment length from inputs in each path

    num_paths_near_max_length = 0
    num_paths_near_min_length = 0

    if max_alignment_length is not None:
        num_paths_near_max_length = sum(1 for length in alignment_lengths if length >= max_alignment_length - 1)

    if min_alignment_length is not None:
        num_paths_near_min_length = sum(1 for length in alignment_lengths if length <= min_alignment_length + 1)

    stats = {
        'num_paths': len(paths),
        'min_cost': min(costs),
        'max_cost': max(costs),
        'avg_cost': mean(costs),
        'mean_alignment_lengths': mean(alignment_lengths),
        'min_alignment_length': min(alignment_lengths),
        'max_alignment_length': max(alignment_lengths),
        'avg_alignment_length': mean(alignment_lengths),
        'num_paths_near_max_length': num_paths_near_max_length,
        'num_paths_near_min_length': num_paths_near_min_length
    }

    return stats

def prune_paths_by_length(paths, max_paths: int = 100_000):
    """
    Prunes the heapq to ensure it contains at most `max_paths` paths using heapq.nsmallest.
    
    Parameters:
    paths -- the priority queue (list of tuples containing (cost, heap_counter, state, path, inputs, remaining_trace_idx, visited))
    max_paths -- maximum number of paths to keep in the heap
    
    Returns:
    A pruned heapq with at most `max_paths` entries.
    """
    if len(paths) > max_paths:
        # Get the top `max_paths` smallest cost elements without sorting everything
        pruned_paths = heapq.nsmallest(max_paths, paths, key=lambda x: x[0])
        # Re-heapify the pruned list
        heapq.heapify(pruned_paths)
        return pruned_paths
    
    return paths

def syncable(state: DfaState, char: int, syncable_dict) -> bool:
    """ 
    Returns True if we can sync the `char` while being in the `state`. It
    returns False otherwise

    Args:
        state: a DfaState object that represents the current state
        char: an integer that represents the current read char

    Returns:
        True if char is syncable, False otherwise.
    """
    return char in syncable_dict[state.state_id]
    

