import json
import warnings
from sys import exception
from typing import Generator, List, Optional

import toml

from alignment.actions import print_action
from alignment.alignment import trace_disalignment
from alignment.utils import postprocess_alignment
from automata_learning.learning import learning_pipeline
from config import ConfigParams
from constants import MAX_LENGTH, cat2id
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, EmptyDatasetError,
                        NoTargetStatesError, SplitNotCoherent)
from generation.dataset.utils import interaction_to_tensor
from generation.utils import equal_ys, labels2cat
from models.utils import pad, topk, trim
from performance_evaluation.alignment.utils import preprocess_interaction
from type_hints import GoodBadDataset, RecDataset, RecModel, SplitTuple
from utils import TimedFunction, seq_tostr
from utils_classes.distances import edit_distance
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
    EmptyDatasetError: "EmptyDatasetError",
}


def parse_splits(splits: Optional[List[tuple]] = None) -> List[Split]:  # type: ignore
    """Parse splits from list of ints to Split object)"""
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
    return splits


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


def run_genetic(
    start_i: int = 0,
    end_i: Optional[int] = None,
    k: int = ConfigParams.TOPK,
    split: Optional[tuple] = None,  # type: ignore
    use_cache: bool = False,
):
    run_log = {
        "i": None,
        "strategy": ConfigParams.GENERATION_STRATEGY,
        "status": None,
        "score": None,
        "source": None,
        "aligned": None,
        "alignment": None,
        "cost": None,
        "gt": None,
        "aligned_gt": None,
        "dataset_time": None,
        "use_cache": use_cache,
    }

    split: Split = parse_splits([split] if split else None)[0]  # type: ignore

    datasets = TimedGenerator(
        DatasetGenerator(
            use_cache=use_cache,
            return_interaction=True,
            limit_generation_to="bad",
            genetic_split=split,
        )
    )
    model = datasets.generator.model  # type: ignore
    i = 0

    if not end_i:
        end_i = 10_000
    for _ in range(start_i):
        datasets.skip()

    for i in range(start_i, end_i):
        try:
            dataset, interaction = next(datasets)
        except EmptyDatasetError as e:
            print(f"Raised {type(e)}")
            run_log["status"] = error_messages[type(e)]
            datasets.skip()
            # Because of the EmptyDatasetError, there may be a mismatch between the dataset indices.
            if datasets.generator.interactions.index != datasets.generator.index:
                _max = max(
                    datasets.generator.interactions.index, datasets.generator.index
                )
                datasets.generator.interactions.index = _max
                datasets.generator.index = _max
            yield run_log
            continue
        except StopIteration:
            return

        # Obtain source categories
        source_sequence = interaction_to_tensor(interaction)  # type: ignore
        gt_preds = model(source_sequence)  # type: ignore
        source_sequence = trim(source_sequence.squeeze(0))
        source_gt = labels2cat(
            topk(logits=gt_preds, k=k, dim=-1, indices=True).squeeze(0), encode=True
        )
        run_log["gt"] = seq_tostr(source_gt)
        if ConfigParams.GENERATION_STRATEGY == "targeted":
            target = {cat2id[t] for t in ConfigParams.TARGET_CAT}  # type: ignore
            target_ys = [target for _ in range(k)]
            run_log["target_y"] = seq_tostr(target_ys)

        run_log["dataset_time"] = datasets.get_times()[i]
        run_log["i"] = i
        run_log["source"] = seq_tostr(source_sequence)

        # if ConfigParams.GENERATION_STRATEGY == "targeted":
        _, counterfactuals = dataset
        # else:
        #     counterfactuals, _ = dataset

        # NOTE: for now I take just the counterfactual which is the most similar to the source sequence, but since they are all counterfactuals,
        # we can also generate a list of different counterfactuals.
        counterfactual, _ = max(
            counterfactuals,
            key=lambda x: -edit_distance(x[0].squeeze(), source_sequence),
        )
        cpreds = model(pad(counterfactual, MAX_LENGTH).unsqueeze(0))
        clabel = labels2cat(
            topk(logits=cpreds, k=k, dim=-1, indices=True).squeeze(0), encode=True
        )
        if not ConfigParams.GENERATION_STRATEGY == "targeted":
            _, score = equal_ys(source_gt, clabel, return_score=True)  # type: ignore
        else:
            _, score = equal_ys(target_ys, clabel, return_score=True)  # type: ignore
        run_log["score"] = (
            score  # if targeted higher is better, otherwise lower is better
        )
        run_log["cost"] = edit_distance(
            counterfactual.squeeze(),
            source_sequence,
            normalized=False,
        )
        if score >= ConfigParams.THRESHOLD:
            run_log["status"] = "good"
        else:
            run_log["status"] = "bad"
        run_log["aligned"] = seq_tostr(trim(counterfactual.squeeze()))
        run_log["aligned_gt"] = seq_tostr(clabel)
        print(json.dumps(run_log, indent=2))
        yield run_log


def run_full(
    dataset_type: RecDataset = ConfigParams.DATASET,
    model_type: RecModel = ConfigParams.MODEL,
    start_i: int = 0,
    end_i: Optional[int] = None,
    splits: Optional[List[tuple]] = None,  # type: ignore
    k: int = ConfigParams.TOPK,
    use_cache: bool = False,
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
    splits = parse_splits(splits)  # type: ignore

    if end_i is None:
        end_i = 10_000
    assert (
        start_i < end_i
    ), f"Start index must be strictly less than end index: {start_i} < {end_i}"

    # Running loop
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

        try:
            dataset, interaction = next(datasets)
        except StopIteration:
            return

        assert (
            len(datasets.get_times()) == i + 1
        ), f"{len(datasets.get_times())} != {i+1}"

        run_log["dataset_time"] = datasets.get_times()[i]
        run_log["i"] = i

        source_sequence: List[Tensor] = preprocess_interaction(interaction)  # type: ignore
        if ConfigParams.GENERATION_STRATEGY != "targeted":
            gt_preds = model(pad(source_sequence, MAX_LENGTH).unsqueeze(0))  # type: ignore

            source_gt = labels2cat(
                topk(logits=gt_preds, k=k, dim=-1, indices=True).squeeze(0), encode=True
            )
        else:
            if not ConfigParams.TARGET_CAT:
                raise ValueError("target must not be None if strategy is 'targeted'")
            source_gt = [
                {cat2id[cat] for cat in ConfigParams.TARGET_CAT} for _ in range(k)
            ]

        # TODO: I don't  think this is useful now, test if run works with the not-categorized dataset
        # if isinstance(datasets.generator.good_strat, CategorizedGeneticStrategy):  # type: ignore
        #     source_gt = set(label2cat(source_gt, encode=True))  # type: ignore

        run_log["source"] = seq_tostr(source_sequence)
        run_log["gt"] = seq_tostr(source_gt)
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
                EmptyDatasetError,
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

            run_log["aligned_gt"] = seq_tostr(aligned_gt)

            is_good, score = equal_ys(source_gt, aligned_gt, return_score=True)  # type: ignore
            if ConfigParams.GENERATION_STRATEGY == "targeted":
                is_good = not is_good
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
