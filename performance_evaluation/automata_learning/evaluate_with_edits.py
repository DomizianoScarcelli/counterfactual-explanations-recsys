from pandas._libs.lib import is_integer
from sklearn.metrics import confusion_matrix
from constants import MAX_LENGTH
from generation.utils import labels2cat
from models.utils import pad
from generation.utils import equal_ys
from models.utils import topk
from alignment.actions import encode_action
from collections import deque
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from aalpy.automata.Dfa import Dfa, DfaState
from utils import printd
import json
import warnings
from pathlib import Path
from typing import Optional, List, Set, Tuple

import fire
import pandas as pd
from aalpy.automata.Dfa import Dfa
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import os
from torch import Tensor, equal, isin
from tqdm import tqdm

from alignment.actions import Action, decode_action
from automata_learning.passive_learning import learning_pipeline
from automata_learning.utils import run_automata
from config import ConfigDict, ConfigParams
from generation.dataset.generate import generate
from generation.dataset.utils import dataset_difference
from models.config_utils import generate_model, get_config
from models.utils import pad_batch, trim
from performance_evaluation.alignment.utils import (
    log_run,
    pk_exists,
    preprocess_interaction,
)
from performance_evaluation.evaluation_utils import (
    compute_metrics,
    print_confusion_matrix,
)
from type_hints import GoodBadDataset, RecModel
from utils import SeedSetter, seq_tostr
from utils_classes.generators import DatasetGenerator


def generate_edits(seq: List[int], dfa_state: DfaState) -> List[Tuple[List[int], bool]]:
    """Given a sequence and a DFA in a certain state, it returns the list of sequences that are reachable with a single step in the dfa."""
    actions = set()
    print(f"[DEBUG] generating edits for seq: ", seq)
    for encoded_edit, state in dfa_state.transitions.items():
        action, item_id = decode_action(encoded_edit)
        print(f"[DEBUG] decoded_edit: ", action, item_id)

        if action == Action.SYNC:
            res = seq + [item_id]
        elif action == Action.DEL and seq[-1] == item_id:
            res = seq[:-1]
        elif action == Action.ADD:
            res = seq + [item_id]
        else:
            continue

        actions.add((res, state.is_accepting))
    print(f"[DEBUG] actions: ", actions)
    return list(actions)


def evaluate_single(
    seq: List[int],
    model: SequentialRecommender,
    dfa_state: DfaState,
    k: int,
    target: Optional[str],
):
    seq_gt = model(pad(torch.tensor(seq), MAX_LENGTH))
    topk_gt = topk(seq_gt, k=k, dim=-1, indices=True)

    edits = generate_edits(seq, dfa_state)
    seq_edits = [edit[0] for edit in edits]
    ys_edits = [edit[1] for edit in edits]
    edits_matrix = torch.tensor(pad_batch(seq_edits, MAX_LENGTH))
    # TODO: maybe do batches
    gts = model(edits_matrix)
    if target is None:

        def score_fn(i: int) -> Tuple[bool, bool]:
            automata_accepts = ys_edits[i]
            model_ys = gts[i]
            model_ys = topk(model_ys, k, dim=-1, indices=True)
            model_accepts = equal_ys(topk_gt, model_ys)
            assert isinstance(model_accepts, bool)
            return (not model_accepts, automata_accepts)

    else:

        def score_fn(i: int) -> Tuple[bool, bool]:
            automata_accepts = ys_edits[i]
            model_ys = gts[i]
            model_ys = topk(model_ys, k, dim=-1, indices=True)
            gt_cats = labels2cat(topk_gt, encode=True)
            edits_cat = labels2cat(model_ys, encode=True)
            model_accepts = equal_ys(gt_cats, edits_cat)
            assert isinstance(model_accepts, bool)
            return (model_accepts, automata_accepts)

    edits_accept = [score_fn(i) for i in range(len(gts))]
    tp, tn, fp, fn = 0, 0, 0, 0
    for model_accepts, automata_accepts in edits_accept:
        if model_accepts and automata_accepts:
            tp += 1
        if not model_accepts and not automata_accepts:
            tn += 1
        if not model_accepts and automata_accepts:
            fp += 1
        if model_accepts and not automata_accepts:
            fn += 1
    precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
    print_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)
    return precision, accuracy, recall


def evaluate_all(
    state: DfaState, trace: Optional[List[int]], target_states: Set[DfaState]
):
    # Perform BFS from the current state to any target state
    queue = deque([(state, 0)])  # Queue of (current_state, hop_length)
    visited = set()
    visited.add(state.state_id)

    while queue:
        current_state, hop_length = queue.popleft()

        if current_state in target_states:
            return hop_length

        # Process transitions for sync, add, and del actions
        for next_state in current_state.transitions.values():
            if next_state.state_id not in visited:
                visited.add(next_state.state_id)
                queue.append((next_state, hop_length + 1))

    return float("inf")  # No target state is reachable


def main():
    config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    model = generate_model(config)
    datasets = DatasetGenerator(
        config=config,
        use_cache=False,
        return_interaction=True,
    )
    start_i = 0
    end_i = 10
    for _ in range(start_i):
        datasets.skip()
    for i in range(start_i, end_i):
        dataset, interaction = next(datasets)
        seq = preprocess_interaction(interaction)
        dfa = learning_pipeline(source=seq, dataset=dataset)
        evaluate(seq=seq, model=model, dfa=dfa)
