from generation.utils import _evaluate_categorized_generation
import warnings
from typing import Any, Dict, List

from recbole.model.abstract_recommender import SequentialRecommender
from torch import Tensor

from alignment.alignment import trace_disalignment
from automata_learning.learning import learning_pipeline
from config import ConfigParams
from constants import MAX_LENGTH, cat2id
from exceptions import (
    CounterfactualNotFound,
    DfaNotAccepting,
    DfaNotRejecting,
    EmptyDatasetError,
    NoTargetStatesError,
    SplitNotCoherent,
)
from generation.utils import equal_ys, labels2cat
from models.utils import pad, topk, trim
from performance_evaluation.genetic.old_evaluate import evaluate_dataset
from type_hints import CategorizedDataset, GoodBadDataset
from utils import TimedFunction, seq_tostr
from utils_classes.distances import edit_distance
from utils_classes.generators import TimedGenerator

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

timed_learning_pipeline = TimedFunction(learning_pipeline)
timed_trace_disalignment = TimedFunction(trace_disalignment)

error_messages = {
    DfaNotAccepting: "DfaNotAccepting",
    DfaNotRejecting: "DfaNotRejecting",
    NoTargetStatesError: "NoTargetStatesError",
    CounterfactualNotFound: "CounterfactualNotFound",
    SplitNotCoherent: "SplitNotCoherent",
    EmptyDatasetError: "EmptyDatasetError",
}


def _init_log(ks: List[int]) -> Dict[str, Any]:
    log_at_ks = [
        {
            f"gen_aligned_gt@{k}": None,
            f"gen_gt@{k}": None,
            f"gen_target_y@{k}": None,
            f"gen_score@{k}": None,
        }
        for k in ks
    ]
    run_log = {
        "i": None,
        "gen_strategy": ConfigParams.GENERATION_STRATEGY,
        "gen_error": None,
        "gen_source": None,
        "gen_aligned": None,
        "gen_alignment": None,
        "gen_cost": None,
        "gen_gt": None,
        "gen_aligned_gt": None,
        "gen_dataset_time": None,
        "gen_good_points_percentage": None,
        "gen_bad_points_percentage": None,
        "gen_good_points_edit_distance": None,
        "gen_bad_points_edit_distance": None,
    }

    for log_at_k in log_at_ks:
        run_log = {**run_log, **log_at_k}
    return run_log


def log_error(error: str, ks: List[int]) -> Dict[str, Any]:
    log = _init_log(ks)
    log["gen_error"] = error
    return log


def evaluate_genetic(
    i: int,
    datasets: TimedGenerator,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    target_cat: List[str],
    ks: List[int],
) -> Dict[str, Any]:
    """Given the ground truth and the preds, it returns a dictionary containing the evaluation metrics."""
    if not ConfigParams.GENERATION_STRATEGY == "targeted":
        raise ValueError(
            f"You are using the `evaluate_targeted` evaluation function but the generation strategy is set to '{ConfigParams.GENERATION_STRATEGY}', change it to 'targeted' or use a different evaluation function"
        )
    log = _init_log(ks)
    source_logits = model(source)
    trimmed_source = trim(source.squeeze(0))
    source_preds = {
        k: labels2cat(
            topk(logits=source_logits, k=k, dim=-1, indices=True).squeeze(0),
            encode=True,
        )
        for k in ks
    }

    target_categories = {cat2id[t] for t in target_cat}  # type: ignore
    target_preds = {k: [target_categories for _ in range(k)] for k in ks}

    # Compute dataset metrics
    if ConfigParams.GENERATION_STRATEGY in ["targeted", "genetic_categorized"]:
        if ConfigParams.GENERATION_STRATEGY == "targeted":
            bad, good = dataset
        else:
            good, bad = dataset
        bad_perc = len(bad) / ConfigParams.POP_SIZE
        good_perc = len(good) / ConfigParams.POP_SIZE
        target_cats = [set(cat2id[cat] for cat in target_cat) for _ in range(min(ks))]
        _, (_, bad_mean_dist) = _evaluate_categorized_generation(
            trimmed_source, bad, target_cats  # type: ignore
        )
        _, (_, good_mean_dist) = _evaluate_categorized_generation(
            trimmed_source, good, target_cats  # type: ignore
        )
        log["gen_good_points_percentage"] = good_perc * 100
        log["gen_bad_points_percentage"] = bad_perc * 100
        log["gen_good_points_edit_distance"] = good_mean_dist
        log["gen_bad_points_edit_distance"] = bad_mean_dist
    # NOTE: for now I take just the counterfactual which is the most similar to the source sequence, but since they are all counterfactuals,
    # we can also generate a list of different counterfactuals.

    _, counterfactuals = dataset
    best_counterfactual, _ = max(
        counterfactuals,
        key=lambda x: -edit_distance(trim(x[0]).squeeze(), trimmed_source),
    )
    counterfactual_logits = model(pad(best_counterfactual, MAX_LENGTH).unsqueeze(0))
    best_counterfactual = trim(best_counterfactual.squeeze())
    counterfactual_preds = {
        k: labels2cat(
            topk(logits=counterfactual_logits, k=k, dim=-1, indices=True).squeeze(0),
            encode=True,
        )
        for k in ks
    }

    for k in ks:
        log[f"gen_target_y@{k}"] = seq_tostr(target_preds[k])
        log[f"gen_gt@{k}"] = seq_tostr(source_preds[k])
        log["gen_dataset_time"] = datasets.get_times()[i]
        log["i"] = i
        log["gen_source"] = seq_tostr(trimmed_source)
        #
        if ConfigParams.GENERATION_STRATEGY != "targeted":
            _, score = equal_ys(
                source_preds[k], counterfactual_preds[k], return_score=True
            )
        else:
            _, score = equal_ys(
                target_preds[k], counterfactual_preds[k], return_score=True
            )
        # if targeted higher is better, otherwise lower is better
        log[f"gen_score@{k}"] = score
        log[f"gen_aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

    log["gen_aligned"] = seq_tostr(best_counterfactual)
    log["gen_cost"] = edit_distance(
        best_counterfactual,
        trimmed_source,
        normalized=False,
    )
    return log
