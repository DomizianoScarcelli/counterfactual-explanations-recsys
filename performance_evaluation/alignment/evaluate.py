from typing import Any, Dict, Generator, List, Optional

from pandas.core.dtypes.common import is_int64_dtype
from recbole.model.abstract_recommender import SequentialRecommender
from torch import Tensor

from alignment.actions import print_action
from alignment.alignment import trace_disalignment
from alignment.utils import postprocess_alignment
from automata_learning.passive_learning import learning_pipeline
from config import ConfigParams
from constants import cat2id, error_messages
from exceptions import (
    CounterfactualNotFound,
    DfaNotAccepting,
    DfaNotRejecting,
    NoTargetStatesError,
    SplitNotCoherent,
)
from generation.utils import equal_ys, labels2cat
from models.utils import topk, trim
from type_hints import CategorySet, GoodBadDataset
from utils import TimedFunction, printd, seq_tostr
from utils_classes.Split import Split

timed_learning_pipeline = TimedFunction(learning_pipeline)
timed_trace_disalignment = TimedFunction(trace_disalignment)


def single_run(
    source_sequence: List[int],
    _dataset: GoodBadDataset,
    split: Optional[Split] = None,
):
    assert isinstance(
        source_sequence, list
    ), f"Source sequence is not a list, but a {type(source_sequence)}"
    assert isinstance(
        source_sequence[0], int
    ), f"Elements of the source sequences are not ints, but {type(source_sequence[0])}"

    dfa = timed_learning_pipeline(source=source_sequence, dataset=_dataset)

    if split is not None:
        source_sequence = split.apply(source_sequence)  # type: ignore

    aligned, cost, alignment = timed_trace_disalignment(dfa, source_sequence)
    aligned = postprocess_alignment(aligned)
    return aligned, cost, alignment


def _init_log(ks: List[int]) -> Dict[str, Any]:

    log_at_ks = [
        {
            f"aligned_gt@{k}": None,
            f"gt@{k}": None,
            f"preds_gt@{k}": None,
            f"score@{k}": None,
        }
        for k in ks
    ]
    run_log = {
        "i": None,
        "split": None,
        "status": None,
        "score": None,
        "source": None,
        "aligned": None,
        "alignment": None,
        "cost": None,
        "gt": None,
        "aligned_gt": None,
        "dataset_time": None,
        "align_time": None,
        "automata_learning_time": None,
    }

    for log_at_k in log_at_ks:
        run_log = {**run_log, **log_at_k}
    return run_log


def log_error(
    i: int, error: str, ks: List[int], split: Split, target_cat: Optional[str] = None
) -> Dict[str, Any]:
    from performance_evaluation.genetic.evaluate import _init_log as gen_init_log

    log = _init_log(ks)
    gen_log = gen_init_log(ks)
    log["i"] = i
    log["split"] = split
    if target_cat:
        log["gen_target_y@1"] = str(
            {cat2id[target_cat]} if isinstance(target_cat, str) else {target_cat}
        )
    gen_log.update(log)
    gen_log.update(ConfigParams.configs_dict())
    gen_log["error"] = error
    return gen_log


def _evaluate_targeted_cat(
    i: int,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    ks: List[int],
    target_cat: str,
    split: Split,
) -> Dict[str, Any]:
    log = _init_log(ks)
    log["i"] = i

    source_logits = model(source)
    trimmed_source = trim(source.squeeze()).tolist()
    source_preds = {
        k: labels2cat(
            topk(logits=source_logits, k=k, dim=-1, indices=True).squeeze(0),
            encode=True,
        )
        for k in ks
    }
    target_categories = {cat2id[target_cat]}
    target_preds = {k: [target_categories for _ in range(k)] for k in ks}

    printd(f"----RUN DEBUG-----")
    printd(f"Current Split: {split}")
    try:
        aligned, cost, alignment = single_run(trimmed_source, dataset, split)

        log["aligned"] = seq_tostr(aligned.squeeze(0).tolist())
        log["alignment"] = seq_tostr([print_action(a) for a in alignment])
        log["cost"] = cost

        log["align_time"] = timed_trace_disalignment.get_last_time()

        counterfactual_logits = model(aligned)
        counterfactual_preds: Dict[int, List[CategorySet]] = {
            k: labels2cat(
                topk(logits=counterfactual_logits, k=k, dim=-1, indices=True).squeeze(
                    0
                ),
                encode=True,
            )
            for k in ks
        }

        for k in ks:
            log[f"gt@{k}"] = seq_tostr(source_preds[k])
            log[f"aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

            # TODO: is this correct?
            if ConfigParams.GENERATION_STRATEGY != "targeted":
                _, score = equal_ys(
                    source_preds[k], counterfactual_preds[k], return_score=True
                )
            else:
                _, score = equal_ys(
                    target_preds[k], counterfactual_preds[k], return_score=True
                )

            log[f"score@{k}"] = score

        log["source"] = seq_tostr(trimmed_source)
        log["split"] = str(split)
    except (
        DfaNotAccepting,
        DfaNotRejecting,
        NoTargetStatesError,
        CounterfactualNotFound,
        SplitNotCoherent,
    ) as e:
        printd(f"run_full: Raised {type(e)}")
        log = log_error(
            i, error=error_messages[type(e)], ks=ks, split=split, target_cat=target_cat
        )
    return log


def _evaluate_targeted_uncat(
    i: int,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    ks: List[int],
    target_cat: str,
    split: Split,
) -> Dict[str, Any]:
    # TODO: this is a copy paste, has to be modified
    log = _init_log(ks)
    log["i"] = i

    source_logits = model(source)
    trimmed_source = trim(source.squeeze()).tolist()
    source_preds = {
        k: topk(logits=source_logits, k=k, dim=-1, indices=True).squeeze(0) for k in ks
    }
    target_preds = {k: [target_cat for _ in range(k)] for k in ks}
    try:
        aligned, cost, alignment = single_run(trimmed_source, dataset, split)

        log["aligned"] = seq_tostr(aligned.squeeze(0).tolist())
        log["alignment"] = seq_tostr([print_action(a) for a in alignment])
        log["cost"] = cost

        log["align_time"] = timed_trace_disalignment.get_last_time()

        counterfactual_logits = model(aligned)
        counterfactual_preds: Dict[int, List[Tensor]] = {
            k: topk(logits=counterfactual_logits, k=k, dim=-1, indices=True).squeeze(0)
            for k in ks
        }

        for k in ks:
            log[f"gt@{k}"] = seq_tostr(source_preds[k])
            log[f"aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

            _, score = equal_ys(
                target_preds[k], counterfactual_preds[k], return_score=True
            )

            log[f"score@{k}"] = score

        log["source"] = seq_tostr(trimmed_source)
        log["split"] = str(split)
    except (
        DfaNotAccepting,
        DfaNotRejecting,
        NoTargetStatesError,
        CounterfactualNotFound,
        SplitNotCoherent,
    ) as e:
        printd(f"run_full: Raised {type(e)}")
        log = log_error(
            i, error=error_messages[type(e)], ks=ks, split=split, target_cat=target_cat
        )
    return log


def _evaluate_untargeted_cat(
    i: int,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    ks: List[int],
    split: Split,
) -> Dict[str, Any]:
    log = _init_log(ks)
    log["i"] = i

    source_logits = model(source)
    trimmed_source = trim(source.squeeze()).tolist()
    source_preds = {
        k: labels2cat(
            topk(logits=source_logits, k=k, dim=-1, indices=True).squeeze(0),
            encode=True,
        )
        for k in ks
    }

    try:
        aligned, cost, alignment = single_run(trimmed_source, dataset, split)

        log["aligned"] = seq_tostr(aligned.squeeze(0).tolist())
        log["alignment"] = seq_tostr([print_action(a) for a in alignment])
        log["cost"] = cost

        log["align_time"] = timed_trace_disalignment.get_last_time()

        counterfactual_logits = model(aligned)
        counterfactual_preds: Dict[int, List[CategorySet]] = {
            k: labels2cat(
                topk(logits=counterfactual_logits, k=k, dim=-1, indices=True).squeeze(
                    0
                ),
                encode=True,
            )
            for k in ks
        }

        for k in ks:
            log[f"gt@{k}"] = seq_tostr(source_preds[k])
            log[f"aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

            _, score = equal_ys(
                source_preds[k], counterfactual_preds[k], return_score=True
            )

            log[f"score@{k}"] = score

        log["source"] = seq_tostr(trimmed_source)
        log["split"] = str(split)
    except (
        DfaNotAccepting,
        DfaNotRejecting,
        NoTargetStatesError,
        CounterfactualNotFound,
        SplitNotCoherent,
    ) as e:
        printd(f"run_full: Raised {type(e)}")
        log = log_error(i, error=error_messages[type(e)], ks=ks, split=split)
    return log
    pass


def _evaluate_untargeted_uncat(
    i: int,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    ks: List[int],
    split: Split,
) -> Dict[str, Any]:
    log = _init_log(ks)
    log["i"] = i

    source_logits = model(source)
    trimmed_source = trim(source.squeeze()).tolist()
    source_preds = {
        k: topk(logits=source_logits, k=k, dim=-1, indices=True).squeeze(0) for k in ks
    }
    try:
        aligned, cost, alignment = single_run(trimmed_source, dataset, split)

        log["aligned"] = seq_tostr(aligned.squeeze(0).tolist())
        log["alignment"] = seq_tostr([print_action(a) for a in alignment])
        log["cost"] = cost

        log["align_time"] = timed_trace_disalignment.get_last_time()

        counterfactual_logits = model(aligned)
        counterfactual_preds: Dict[int, List[CategorySet]] = {
            k: topk(logits=counterfactual_logits, k=k, dim=-1, indices=True).squeeze(0)
            for k in ks
        }

        for k in ks:
            log[f"gt@{k}"] = seq_tostr(source_preds[k])
            log[f"aligned_gt@{k}"] = seq_tostr(counterfactual_preds[k])

            _, score = equal_ys(
                source_preds[k], counterfactual_preds[k], return_score=True
            )

            log[f"score@{k}"] = score

        log["source"] = seq_tostr(trimmed_source)
        log["split"] = str(split)
    except (
        DfaNotAccepting,
        DfaNotRejecting,
        NoTargetStatesError,
        CounterfactualNotFound,
        SplitNotCoherent,
    ) as e:
        printd(f"run_full: Raised {type(e)}")
        log = log_error(i, error=error_messages[type(e)], ks=ks, split=split)
    return log


def evaluate_alignment(
    i: int,
    dataset: GoodBadDataset,
    source: Tensor,
    model: SequentialRecommender,
    ks: List[int],
    target_cat: Optional[str],
    split: Split,
) -> Dict[str, Any]:
    """Given the ground truth and the preds, it returns a dictionary containing the evaluation metrics."""
    categorized = ConfigParams.CATEGORIZED
    if target_cat is not None and categorized:
        return _evaluate_targeted_cat(
            i=i,
            dataset=dataset,
            source=source,
            model=model,
            target_cat=target_cat,
            ks=ks,
            split=split,
        )
    if target_cat is not None and not categorized:
        return _evaluate_targeted_uncat(
            i=i,
            dataset=dataset,
            source=source,
            model=model,
            target_cat=target_cat,
            ks=ks,
            split=split,
        )
    if target_cat is None and categorized:
        return _evaluate_untargeted_uncat(
            i=i, dataset=dataset, source=source, model=model, ks=ks, split=split
        )
    if target_cat is None and not categorized:
        return _evaluate_untargeted_uncat(
            i=i, dataset=dataset, source=source, model=model, ks=ks, split=split
        )
    else:
        raise ValueError()
