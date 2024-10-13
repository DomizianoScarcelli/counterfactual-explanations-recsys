from aalpy.automata.Dfa import Dfa, DfaState
from automata_utils import invert_automata
from dataset_generator import NumItems
from tqdm import tqdm
from typing import List
from automata_utils import run_automata
from graph_search import (decode_action, get_shortest_alignment_dijkstra, 
                          get_shortest_alignment_a_star)
from copy import deepcopy
from graph_search import Action, decode_action, act_str


def evaluate_trace_disalignment(interactions: Tuple[torch.Tensor, int], 
                                oracle: Callable[[torch.Tensor], torch.Tensor]) -> float:
    """
    Given a dataset of interactions , it performs disalignment (aka
                                                                counterfactual
                                                                generation)
    which results in a dataset counterfactuals. For each couple (i, not_i) in
    zip(interactions, counterfactuals), it computes the percentage of examples
    where model(i) != model(not_i). Model is a black box recommender system
    (aka oracle).

    Args:
        interaction: [TODO:description]
        oracle: [TODO:description]
    Returns:
        the percentace of correct counterfactuals 


    """
    pass

