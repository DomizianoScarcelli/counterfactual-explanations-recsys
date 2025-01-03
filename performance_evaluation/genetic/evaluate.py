import warnings
from typing import Any, Dict, List

from recbole.model.abstract_recommender import SequentialRecommender
from torch import Tensor

from alignment.alignment import trace_disalignment
from automata_learning.learning import learning_pipeline
from config import ConfigParams
from constants import MAX_LENGTH, cat2id
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, EmptyDatasetError,
                        NoTargetStatesError, SplitNotCoherent)
from generation.utils import equal_ys, labels2cat
from models.utils import pad, topk, trim
from type_hints import CategorySet, Dataset
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
            f"aligned_gt@{k}": None,
            f"gt@{k}": None,
            f"target_y@{k}": None,
        }
        for k in ks
    ]
    run_log = {
        "i": None,
        "strategy": ConfigParams.GENERATION_STRATEGY,
        "error": None,
        "score": None,
        "source": None,
        "aligned": None,
        "alignment": None,
        "cost": None,
        "gt": None,
        "aligned_gt": None,
        "dataset_time": None,
    }

    for log_at_k in log_at_ks:
        run_log = {**run_log, **log_at_k}
    return run_log


def log_error(error: str, ks: List[int]) -> Dict[str, Any]:
    log = _init_log(ks)
    log["error"] = error
    return log


def evaluate_targeted(
    i: int,
    datasets: TimedGenerator,
    source: Tensor,
    counterfactuals: Dataset,
    model: SequentialRecommender,
    ks: List[int],
):
    """Given the ground truth and the preds, it returns a dictionary containing the evaluation metrics."""
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

    target_categories = {cat2id[t] for t in ConfigParams.TARGET_CAT}  # type: ignore
    target_preds = {k: [target_categories for _ in range(k)] for k in ks}

    # NOTE: for now I take just the counterfactual which is the most similar to the source sequence, but since they are all counterfactuals,
    # we can also generate a list of different counterfactuals.
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
        log[f"target_y@{k}"] = seq_tostr(target_preds[k])
        log[f"gt@{k}"] = seq_tostr(source_preds[k])
        log["dataset_time"] = datasets.get_times()[i]
        log["i"] = i
        log["source"] = seq_tostr(trimmed_source)
        # NOTE: this is needed if we want to evaluate the genetic algorithm not on the targeted part.
        # if not ConfigParams.GENERATION_STRATEGY == "targeted":
        #     _, score = equal_ys(source_gt[k], clabel[k], return_score=True)  # type: ignore
        # else:

        _, score = equal_ys(target_preds[k], counterfactual_preds[k], return_score=True)
        # if targeted higher is better, otherwise lower is better
        log[f"score@{k}"] = score
        log[f"aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

    log["aligned"] = seq_tostr(best_counterfactual)
    log["cost"] = edit_distance(
        best_counterfactual,
        trimmed_source,
        normalized=False,
    )
    return log
