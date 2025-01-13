from utils import printd
from typing import Any, Dict, Generator, List, Optional

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
from utils import TimedFunction, seq_tostr
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
    i: int, error: str, ks: List[int], split: Split, target_cat: str
) -> Dict[str, Any]:
    from performance_evaluation.genetic.evaluate import _init_log as gen_init_log

    log = _init_log(ks)
    gen_log = gen_init_log(ks)
    log["i"] = i
    log["split"] = split
    log["gen_target_y@1"] = str({cat2id[target_cat]})
    gen_log.update(log)
    gen_log.update(ConfigParams.configs_dict())
    gen_log["error"] = error
    return gen_log


def evaluate_alignment(
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

    if not ConfigParams.GENERATION_STRATEGY == "targeted":
        raise ValueError(
            f"You are using the `evaluate_targeted` evaluation function but the generation strategy is set to '{ConfigParams.GENERATION_STRATEGY}', change it to 'targeted' or use a different evaluation function"
        )

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
