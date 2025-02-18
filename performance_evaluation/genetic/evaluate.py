import warnings
from typing import Any, Dict, List, Optional, Set

from recbole.model.abstract_recommender import SequentialRecommender
from torch import Tensor

from alignment.alignment import trace_disalignment
from automata_learning.passive_learning import learning_pipeline
from config import ConfigParams
from constants import MAX_LENGTH, cat2id
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, EmptyDatasetError,
                        NoTargetStatesError, SplitNotCoherent)
from generation.utils import (_evaluate_categorized_generation,
                              _evaluate_generation, equal_ys, labels2cat)
from models.utils import pad, topk, trim
from type_hints import GoodBadDataset
from utils import TimedFunction, seq_tostr
from utils_classes.distances import edit_distance
from utils_classes.generators import TimedGenerator
from utils_classes.Split import Split

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
            f"gen_source_score@{k}": None,
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


def log_error(
    i: int, error: str, ks: List[int], split: Split, target_cat: Optional[str]
) -> Dict[str, Any]:
    from performance_evaluation.alignment.evaluate import \
        _init_log as align_init_log

    log = _init_log(ks)
    align_log = align_init_log(ks)
    log["i"] = i
    log["gen_error"] = error
    log["split"] = str(split)
    if target_cat:
        log["gen_target_y@1"] = str({cat2id[target_cat]})
    align_log.update(log)

    return align_log


def compute_metrics_log(
    log: Dict[str, Any],
    gt_preds: Dict[int, List[int] | List[Set[int]]],
    source_preds: Dict[int, List[int] | List[Set[int]]],
    counterfactual_preds: Dict[int, List[int] | List[Set[int]]],
):
    ks = gt_preds.keys()
    for k in ks:
        log[f"gen_target_y@{k}"] = seq_tostr(gt_preds[k])
        log[f"gen_gt@{k}"] = seq_tostr(source_preds[k])
        #
        _, source_score = equal_ys(target_preds[k], source_preds[k], return_score=True)
        _, counter_score = equal_ys(
            target_preds[k], counterfactual_preds[k], return_score=True
        )
        # if targeted higher is better, otherwise lower is better
        log[f"gen_score@{k}"] = counter_score
        log[f"gen_source_score@{k}"] = source_score
        log[f"gen_aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

    pass


def _evaluate_targeted_cat(
    i: int,
    datasets: TimedGenerator,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    target_cat: str,
    ks: List[int],
):
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

    target_categories = {cat2id[target_cat]}
    target_preds = {k: [target_categories for _ in range(k)] for k in ks}

    # Compute dataset metrics
    if ConfigParams.GENERATION_STRATEGY in ["targeted", "genetic_categorized"]:
        good, bad = dataset
        bad_perc = len(bad) / ConfigParams.POP_SIZE
        good_perc = len(good) / ConfigParams.POP_SIZE
        target_cats = [{cat2id[target_cat]} for _ in range(min(ks))]
        if len(bad) != 0:
            _, (_, bad_mean_dist) = _evaluate_categorized_generation(
                trimmed_source, bad, target_cats  # type: ignore
            )
            log["gen_bad_points_edit_distance"] = bad_mean_dist
        if len(good) != 0:
            _, (_, good_mean_dist) = _evaluate_categorized_generation(
                trimmed_source, good, target_cats  # type: ignore
            )
            log["gen_good_points_edit_distance"] = good_mean_dist

        log["gen_good_points_percentage"] = good_perc * 100
        log["gen_bad_points_percentage"] = bad_perc * 100
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
        _, source_score = equal_ys(target_preds[k], source_preds[k], return_score=True)
        _, counter_score = equal_ys(
            target_preds[k], counterfactual_preds[k], return_score=True
        )
        # if targeted higher is better, otherwise lower is better
        log[f"gen_score@{k}"] = counter_score
        log[f"gen_source_score@{k}"] = source_score
        log[f"gen_aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

    log["gen_aligned"] = seq_tostr(best_counterfactual)
    log["gen_cost"] = edit_distance(
        best_counterfactual,
        trimmed_source,
        normalized=False,
    )
    return log


def _evaluate_untargeted_cat(
    i: int,
    datasets: TimedGenerator,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    ks: List[int],
):
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

    # Compute dataset metrics not implemented for untargeted categorized
    log["gen_good_points_percentage"] = None
    log["gen_bad_points_percentage"] = None
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
        log[f"gen_target_y@{k}"] = None
        log[f"gen_gt@{k}"] = seq_tostr(source_preds[k])
        log["gen_dataset_time"] = datasets.get_times()[i]
        log["i"] = i
        log["gen_source"] = seq_tostr(trimmed_source)
        #
        _, counter_score = equal_ys(
            source_preds[k], counterfactual_preds[k], return_score=True
        )
        # if targeted higher is better, otherwise lower is better
        log[f"gen_score@{k}"] = counter_score
        log[f"gen_source_score@{k}"] = None
        log[f"gen_aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

    log["gen_aligned"] = seq_tostr(best_counterfactual)
    log["gen_cost"] = edit_distance(
        best_counterfactual,
        trimmed_source,
        normalized=False,
    )
    return log


def _evaluate_untargeted_uncat(
    i: int,
    datasets: TimedGenerator,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    ks: List[int],
):
    log = _init_log(ks)
    source_logits = model(source)
    trimmed_source = trim(source.squeeze(0))
    source_preds = {
        k: topk(logits=source_logits, k=k, dim=-1, indices=True).squeeze(0) for k in ks
    }

    # Compute dataset metrics not implemented for untargeted uncategorized
    log["gen_good_points_percentage"] = None
    log["gen_bad_points_percentage"] = None

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
        k: topk(logits=counterfactual_logits, k=k, dim=-1, indices=True).squeeze(0)
        for k in ks
    }

    for k in ks:
        log[f"gen_target_y@{k}"] = None
        log[f"gen_gt@{k}"] = seq_tostr(source_preds[k])
        log["gen_dataset_time"] = datasets.get_times()[i]
        log["i"] = i
        log["gen_source"] = seq_tostr(trimmed_source)

        _, counter_score = equal_ys(
            source_preds[k], counterfactual_preds[k], return_score=True
        )
        assert (
            0.0 <= counter_score <= 1
        ), f"gen_score@{k} is out of range: {counter_score}"
        # if targeted higher is better, otherwise lower is better
        log[f"gen_score@{k}"] = counter_score
        log[f"gen_source_score@{k}"] = None
        log[f"gen_aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

    log["gen_aligned"] = seq_tostr(best_counterfactual)
    log["gen_cost"] = edit_distance(
        best_counterfactual,
        trimmed_source,
        normalized=False,
    )
    return log


def _evaluate_targeted_uncat(
    i: int,
    datasets: TimedGenerator,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    target_cat: str,
    ks: List[int],
):
    log = _init_log(ks)
    source_logits = model(source)
    trimmed_source = trim(source.squeeze(0))
    source_preds = {
        k: topk(logits=source_logits, k=k, dim=-1, indices=True).squeeze(0) for k in ks
    }

    target_preds = {k: [target_cat for _ in range(k)] for k in ks}

    # Compute dataset metrics not implemented for targeted uncategorized
    log["gen_good_points_percentage"] = None
    log["gen_bad_points_percentage"] = None

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
        k: topk(logits=counterfactual_logits, k=k, dim=-1, indices=True).squeeze(0)
        for k in ks
    }

    for k in ks:
        log[f"gen_target_y@{k}"] = seq_tostr(target_preds[k])
        log[f"gen_gt@{k}"] = seq_tostr(source_preds[k])
        log["gen_dataset_time"] = datasets.get_times()[i]
        log["i"] = i
        log["gen_source"] = seq_tostr(trimmed_source)
        #
        _, source_score = equal_ys(target_preds[k], source_preds[k], return_score=True)
        _, counter_score = equal_ys(
            target_preds[k], counterfactual_preds[k], return_score=True
        )
        # if targeted higher is better, otherwise lower is better
        log[f"gen_score@{k}"] = counter_score
        log[f"gen_source_score@{k}"] = source_score
        log[f"gen_aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

    log["gen_aligned"] = seq_tostr(best_counterfactual)
    log["gen_cost"] = edit_distance(
        best_counterfactual,
        trimmed_source,
        normalized=False,
    )
    return log


def evaluate_genetic(
    i: int,
    datasets: TimedGenerator,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    target_cat: Optional[str],
    ks: List[int],
) -> Dict[str, Any]:
    """Given the ground truth and the preds, it returns a dictionary containing the evaluation metrics."""
    categorized = ConfigParams.CATEGORIZED
    if target_cat is not None and categorized:
        return _evaluate_targeted_cat(
            i=i,
            datasets=datasets,
            dataset=dataset,
            source=source,
            model=model,
            target_cat=target_cat,
            ks=ks,
        )
    if target_cat is not None and not categorized:
        return _evaluate_targeted_uncat(
            i=i,
            datasets=datasets,
            dataset=dataset,
            source=source,
            model=model,
            target_cat=target_cat,
            ks=ks,
        )
    if target_cat is None and categorized:
        return _evaluate_untargeted_cat(
            i=i,
            datasets=datasets,
            dataset=dataset,
            source=source,
            model=model,
            ks=ks,
        )
    if target_cat is None and not categorized:
        return _evaluate_untargeted_uncat(
            i=i,
            datasets=datasets,
            dataset=dataset,
            source=source,
            model=model,
            ks=ks,
        )
    else:
        raise ValueError()
