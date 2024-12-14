from models.utils import pad
from constants import MAX_LENGTH
from generation.utils import labels2cat
from models.utils import topk
import json
import warnings
from typing import Generator, List, Optional

import toml

from alignment.actions import print_action
from alignment.alignment import trace_disalignment
from alignment.utils import postprocess_alignment
from automata_learning.learning import learning_pipeline
from config import ConfigDict, ConfigParams
from exceptions import (
    CounterfactualNotFound,
    DfaNotAccepting,
    DfaNotRejecting,
    NoTargetStatesError,
    SplitNotCoherent,
)
from generation.strategies.genetic_categorized import CategorizedGeneticStrategy
from generation.utils import equal_ys, label2cat
from performance_evaluation.alignment.utils import preprocess_interaction
from type_hints import GoodBadDataset, RecDataset, RecModel, SplitTuple
from utils import TimedFunction, seq_tostr
from utils_classes.generators import DatasetGenerator, TimedGenerator
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
}


def single_run(
    source_sequence: List[int], _dataset: GoodBadDataset, split: Optional[Split] = None
):
    assert isinstance(
        source_sequence, list
    ), f"Source sequence is not a list, but a {type(source_sequence)}"
    assert isinstance(
        source_sequence[0], int
    ), f"Elements of the source sequences are not ints, but {type(source_sequence[0])}"

    dfa = timed_learning_pipeline(source=source_sequence, dataset=_dataset)

    if split:
        source_sequence = split.apply(source_sequence)  # type: ignore

    aligned, cost, alignment = timed_trace_disalignment(dfa, source_sequence)
    aligned = postprocess_alignment(aligned)
    return aligned, cost, alignment


def run(
    dataset_type: RecDataset = ConfigParams.DATASET,
    model_type: RecModel = ConfigParams.MODEL,
    start_i: int = 0,
    end_i: Optional[int] = None,
    splits: Optional[List[int]] = None,  # type: ignore
    k: int = ConfigParams.TOPK,
    use_cache: bool = True,
) -> Generator:

    params = (
        {
            "parameters": {
                "use_cache": use_cache,
                "start_i": start_i,
                "end_i": start_i + 1 if end_i is None else end_i,
                "splits": (
                    "(0, None, 0)"
                    if not splits
                    else ", ".join([str(s) for s in splits])
                ),
            },
            "model": {
                "dataset_type": dataset_type.value,
                "model_type": model_type.value,
            },
        },
    )
    print(
        f"""
-----------------------
CONFIG
-----------------------
---Inputs---
{json.dumps(params, indent=2)}
---Config.toml---
{json.dumps(toml.load(ConfigParams._config_path), indent=2)}
-----------------------
"""
    )

    # Init config
    datasets = TimedGenerator(
        DatasetGenerator(use_cache=use_cache, return_interaction=True)
    )
    model = datasets.generator.model  # type: ignore

    # Parse args
    if splits:
        splits: SplitTuple = tuple(splits)  # type: ignore
        if isinstance(splits[0], tuple):
            splits: List[Split] = [Split(*s) for s in splits]  # type: ignore
        elif (
            isinstance(splits[0], int)
            or isinstance(splits[0], float)
            or splits[0] is None
        ):
            splits: List[Split] = [Split(*splits)]  # type: ignore

        assert isinstance(splits, list) and isinstance(
            splits[0], Split
        ), f"Malformed splits: {splits}"
    else:
        splits = [Split(0, None, 0)]

    if end_i is None:
        end_i = start_i + 1
    assert (
        start_i < end_i
    ), f"Start index must be strictly less than end index: {start_i} < {end_i}"

    # Running loop
    run_log = {
        "i": None,
        "split": None,
        "status": None,
        "source": None,
        "aligned": None,
        "alignment": None,
        "cost": None,
        "gt": None,
        "aligned_gt": None,
        "score": None,
        "dataset_time": None,
        "align_time": None,
        "automata_learning_time": None,
        "use_cache": use_cache,
    }
    i = 0
    while i < end_i:
        # Execute only the loops where start_i <= i < end_i
        if i < start_i:
            print(f"Skipping i = {i}")
            datasets.skip()
            i += 1
            continue

        assert datasets.index == i, f"{datasets.index} != {i}"
        assert len(datasets.get_times()) == i, f"{len(datasets.get_times())} != {i}"

        dataset, interaction = next(datasets)

        assert (
            len(datasets.get_times()) == i + 1
        ), f"{len(datasets.get_times())} != {i+1}"

        run_log["dataset_time"] = datasets.get_times()[i]
        run_log["i"] = i

        source_sequence: List[Tensor] = preprocess_interaction(interaction)  # type: ignore
        gt_preds = model(pad(source_sequence, MAX_LENGTH).unsqueeze(0))  # type: ignore

        source_gt = labels2cat(
            topk(logits=gt_preds, k=k, dim=-1, indices=True).squeeze(0), encode=True
        )

        # TODO: I don't  think this is useful now, test if run works with the not-categorized dataset
        # if isinstance(datasets.generator.good_strat, CategorizedGeneticStrategy):  # type: ignore
        #     source_gt = set(label2cat(source_gt, encode=True))  # type: ignore

        run_log["source"] = seq_tostr(source_sequence)
        run_log["gt"] = source_gt if isinstance(source_gt, int) else str(source_gt)
        for split in splits:
            run_log["split"] = str(split)
            split = split.parse_nan(source_sequence)
            print(f"----RUN DEBUG-----")
            print(f"Current Split: {split}")
            try:
                aligned, cost, alignment = single_run(source_sequence, dataset, split)
            except (
                DfaNotAccepting,
                DfaNotRejecting,
                NoTargetStatesError,
                CounterfactualNotFound,
                SplitNotCoherent,
            ) as e:
                print(f"Raised {type(e)}")
                run_log["status"] = error_messages[type(e)]
                yield run_log
                continue

            run_log["aligned"] = seq_tostr(aligned.squeeze(0).tolist())
            run_log["alignment"] = seq_tostr([print_action(a) for a in alignment])
            run_log["cost"] = cost
            print(f"[{i}] Alignment cost: {cost}")

            run_log["align_time"] = timed_trace_disalignment.get_last_time()
            run_log["automata_learning_time"] = timed_learning_pipeline.get_last_time()

            aligned_gt = labels2cat(
                topk(logits=model(aligned), k=k, dim=-1, indices=True).squeeze(0),
                encode=True,
            )

            run_log["aligned_gt"] = (
                aligned_gt if isinstance(aligned_gt, int) else str(aligned_gt)
            )

            is_good, score = equal_ys(source_gt, aligned_gt, return_score=True)  # type: ignore
            run_log["score"] = score
            if is_good:
                run_log["status"] = "bad"
                print(
                    f"[{i}] Bad counterfactual! {source_gt} == {aligned_gt}, score is {score}"
                )
            else:
                run_log["status"] = "good"
                print(
                    f"[{i}] Good counterfactual! {source_gt} != {aligned_gt}, score is {score}"
                )
            print("--------------------")

            print(json.dumps(run_log, indent=2))
            yield run_log

        i += 1
