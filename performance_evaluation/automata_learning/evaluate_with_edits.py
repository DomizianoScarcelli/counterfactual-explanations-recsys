from constants import MAX_LENGTH
from statistics import mean
from generation.utils import labels2cat
from models.utils import pad
from generation.utils import equal_ys
from models.utils import topk
from collections import deque
from recbole.model.abstract_recommender import SequentialRecommender
from aalpy.automata.Dfa import DfaState
from typing import Optional, List, Set, Tuple

from aalpy.automata.Dfa import DfaState
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from alignment.actions import Action, decode_action
from automata_learning.passive_learning import learning_pipeline
from config import ConfigParams
from models.config_utils import generate_model, get_config
from models.utils import pad_batch
from performance_evaluation.alignment.utils import (
    preprocess_interaction,
)
from performance_evaluation.evaluation_utils import (
    compute_metrics,
    print_confusion_matrix,
)
from utils_classes.generators import DatasetGenerator


def generate_edits(seq: List[int], dfa_state: DfaState) -> List[Tuple[List[int], bool]]:
    """Given a sequence and a DFA in a certain state, it returns the list of sequences that are reachable with a single step in the dfa."""
    actions = []
    for encoded_edit, state in dfa_state.transitions.items():
        action, char = decode_action(encoded_edit)

        if action == Action.SYNC:
            new_seq = seq + [char]
        elif action == Action.DEL and len(seq) > 0 and seq[-1] == char:
            new_seq = seq[:-1]
        elif action == Action.ADD:
            new_seq = seq + [char]
        else:
            continue

        if (new_seq, state.is_accepting) not in actions and new_seq != []:
            actions.append((new_seq, state.is_accepting))
    return actions


def evaluate_single(
    seq: List[int],
    model: SequentialRecommender,
    dfa_state: DfaState,
    k: int,
    target: Optional[str],
):
    seq_gt = model(pad(seq, MAX_LENGTH).unsqueeze(0))
    topk_gt = topk(seq_gt, k=k, dim=-1, indices=True).squeeze()

    edits = generate_edits(seq, dfa_state)
    seq_edits = [edit[0] for edit in edits]
    ys_edits = [edit[1] for edit in edits]
    edits_matrix = pad_batch(seq_edits, MAX_LENGTH)
    # TODO: maybe do batches
    gts = model(edits_matrix)
    edits_matrix = edits_matrix.squeeze()
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
            model_ys = topk(model_ys, k, dim=-1, indices=True).squeeze()
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
    model: SequentialRecommender,
    k: int,
    state: DfaState,
    target_states: Set[DfaState],
    target: Optional[str] = None,
):
    # Perform BFS from the current state to any target state
    queue = deque([(state, 0, [])])  # Queue of (current_state, hop_length, sequence)
    visited = set()
    precision, accuracy, recall = [], [], []
    pbar = tqdm(desc=f"Evaluating all from state {state.state_id}")
    prec_mean, acc_mean, rec_mean = 0, 0, 0

    while queue:
        pbar.update(1)
        current_state, hop_length, seq = queue.popleft()

        # Evaluate the current sequence
        if len(seq) > 1:
            prec, acc, rec = evaluate_single(
                seq=seq, model=model, dfa_state=current_state, k=k, target=target
            )
            precision.append(prec)
            accuracy.append(acc)
            recall.append(rec)
            prec_mean = round(mean(precision), 2) if len(precision) > 0 else 0
            acc_mean = round(mean(accuracy), 2) if len(accuracy) > 0 else 0
            rec_mean = round(mean(recall), 2) if len(recall) > 0 else 0
            pbar.set_postfix_str(f"prec: {prec_mean}, acc: {acc_mean}, rec: {rec_mean}")

        # Process transitions for sync, add, and del actions
        for edge, next_state in current_state.transitions.items():
            if len(seq) >= MAX_LENGTH:
                continue

            if (tuple(seq), edge) not in visited:
                action, char = decode_action(edge)
                visited.add((tuple(seq), edge))

                if action == Action.SYNC:
                    new_seq = seq + [char]
                # elif action == Action.DEL and len(seq) > 0 and seq[-1] == char:
                #     new_seq = seq[:-1]
                # elif action == Action.ADD:
                #     new_seq = seq + [char]
                else:
                    continue

                queue.append((next_state, hop_length + 1, new_seq))

    return prec_mean, acc_mean, rec_mean


def main():
    config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    model = generate_model(config)
    target = "Action"
    datasets = DatasetGenerator(
        config=config, use_cache=False, return_interaction=True, target=target
    )
    start_i = 0
    end_i = 10
    for _ in range(start_i):
        datasets.skip()
    for i in tqdm(range(start_i, end_i), desc="Evaluating automata learning..."):
        dataset, interaction = next(datasets)
        seq = preprocess_interaction(interaction)
        dfa = learning_pipeline(source=seq, dataset=dataset)
        # final_states = {s for s in dfa.states if s.is_accepting}
        final_states = set()
        prec, acc, rec = evaluate_all(
            model=model,
            k=5,
            state=dfa.initial_state,
            target_states=final_states,
            target=target,
        )
        print(f"-" * 50)
        print(f"Prec: {prec}, Acc: {acc}, Rec: {rec}")
        print(f"-" * 50)


if __name__ == "__main__":
    main()
