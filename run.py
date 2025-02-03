import warnings
from ctypes import alignment
from time import strftime
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from config import ConfigParams
from constants import cat2id, error_messages
from exceptions import EmptyDatasetError
from generation.dataset.utils import interaction_to_tensor
from performance_evaluation.alignment.evaluate import evaluate_alignment
from performance_evaluation.alignment.evaluate import log_error as log_alignment_error
from performance_evaluation.alignment.utils import pk_exists
from performance_evaluation.genetic.evaluate import evaluate_genetic
from performance_evaluation.genetic.evaluate import log_error as log_genetic_error
from type_hints import SplitTuple
from utils import printd
from utils_classes.generators import (
    DatasetGenerator,
    InteractionGenerator,
    TimedGenerator,
)
from utils_classes.Split import Split

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def skip_sequence(
    i: int,
    target_cat: Optional[str],
    prev_df: Optional[DataFrame],
    split: tuple,
    alignment: bool = True,
):
    if prev_df is not None:
        new_df = {
            "i": [i],
        }
        primary_key = ["i"]
        if target_cat:
            new_df["gen_target_y@1"] = str(
                {cat2id[target_cat]} if isinstance(target_cat, str) else {target_cat}
            )
            primary_key.append("gen_target_y@1")
        if alignment:
            new_df["split"] = str(split)
            primary_key.append("split")
        new_df.update(ConfigParams.configs_dict())

        future_df = pd.concat(
            [
                prev_df,
                pd.DataFrame(new_df),
            ]
        )
        if pk_exists(
            future_df,
            primary_key=primary_key,
            consider_config=True,
        ):
            return True
    return False


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


def run_genetic(
    target_cat: Optional[str],
    start_i: int = 0,
    end_i: int = 1,
    ks: List[int] = ConfigParams.TOPK,
    split: Optional[tuple] = None,  # type: ignore
    prev_df: Optional[DataFrame] = None,
    pbar=None,
):
    split: Split = parse_splits([split] if split else None)[0]  # type: ignore
    datasets = TimedGenerator(
        DatasetGenerator(
            use_cache=False,
            return_interaction=True,
            limit_generation_to="bad",
            genetic_split=split,
            target=target_cat,
        )
    )
    model = datasets.generator.model  # type: ignore

    for _ in range(start_i):
        datasets.skip()

    for i in range(start_i, end_i):
        if skip_sequence(i, target_cat, prev_df, split.split, alignment=False):
            printd(
                f"Skipping i: {i} with target {target_cat} and split {split} because already in the log..."
            )
            if pbar:
                pbar.total -= 1
                pbar.refresh()
            datasets.skip()
            continue
        try:
            dataset, interaction = next(datasets)
        except EmptyDatasetError as e:
            printd(f"run_genetic: Raised {type(e)}")
            log = log_genetic_error(
                i,
                error=error_messages[type(e)],
                ks=ks,
                split=split,
                target_cat=target_cat,
            )
            datasets.skip()
            datasets.generator.match_indices()  # type: ignore
            yield log
            if pbar:
                pbar.update(1)
            continue
        except StopIteration:
            return

        # Obtain source categories
        source_sequence = interaction_to_tensor(interaction)  # type: ignore

        if pbar:
            pbar.set_postfix_str(f"| Split: {split}")
            pbar.update(1)

        log = evaluate_genetic(
            i=i,
            target_cat=target_cat,
            datasets=datasets,
            source=source_sequence,
            dataset=dataset,
            model=model,
            ks=ks,
        )

        yield log


# TODO: add the possibility for the target_cat to be None, i.e. the run to be untargeted.
def run_alignment(
    target_cat: Optional[str],
    start_i: int = 0,
    end_i: int = 1,
    splits: Optional[List[tuple]] = None,  # type: ignore
    ks: List[int] = ConfigParams.TOPK,
    use_cache: bool = False,
    prev_df: Optional[DataFrame] = None,
    pbar=None,
) -> Generator:

    # Parse args
    splits: List[Split] = parse_splits(splits)

    # Init config
    datasets = TimedGenerator(
        DatasetGenerator(
            use_cache=use_cache,
            return_interaction=True,
            genetic_split=splits[0],
            target=target_cat,
        )
    )
    model = datasets.generator.model  # type: ignore

    assert splits is not None

    assert (
        start_i < end_i
    ), f"Start index must be strictly less than end index: {start_i} < {end_i}"

    for i in range(start_i):
        printd(f"Skipping i = {i}")
        datasets.skip()
        continue
    for i in tqdm(range(start_i, end_i), disable=ConfigParams.DEBUG == 0):
        new_splits = []
        for split in splits:
            if not skip_sequence(i, target_cat, prev_df, split.split):
                new_splits.append(split)
            else:
                if pbar:
                    pbar.total -= 1
                    pbar.refresh()
                printd(
                    f"Skipping i: {i} with target {target_cat} and split {split} because already in the log..."
                )
        if len(new_splits) == 0:
            datasets.skip()
            continue
        splits = new_splits
        assert datasets.index == i, f"{datasets.index} != {i}"
        assert len(datasets.get_times()) == i, f"{len(datasets.get_times())} != {i}"

        try:
            dataset, interaction = next(datasets)
        except StopIteration:
            printd(f"STOP ITERATION RAISED")
            return
        except EmptyDatasetError as e:
            printd(f"run_full: Raised {type(e)}")
            logs = []
            for split in splits:
                log = log_alignment_error(
                    i,
                    error=error_messages[type(e)],
                    ks=ks,
                    split=split,
                    target_cat=target_cat,
                )
                logs.append(log)
            datasets.skip()
            datasets.generator.match_indices()  # type: ignore
            yield logs
            if pbar:
                pbar.update(1)
            continue

        assert (
            len(datasets.get_times()) == i + 1
        ), f"{len(datasets.get_times())} != {i+1}"

        source_sequence = interaction_to_tensor(interaction)  # type: ignore

        alignment_logs = []
        for split in splits:
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"| Split: {split}")
            alignment_log = evaluate_alignment(
                i=i,
                target_cat=target_cat,
                dataset=dataset,
                source=source_sequence,
                model=model,
                ks=ks,
                split=split,  # type: ignore
            )

            alignment_logs.append(alignment_log)

        yield alignment_logs


# TODO:
# - put a boolean 'evaluate' which if false doesn't run the evaluation but just the generation
def run_all(
    target_cat: Optional[str],
    start_i: int = 0,
    end_i: int = 1,
    splits: Optional[List[tuple]] = None,  # type: ignore
    ks: List[int] = ConfigParams.TOPK,
    use_cache: bool = False,
    prev_df: Optional[DataFrame] = None,
    pbar=None,
) -> Generator[List[Dict[str, Any]], None, None]:

    # Parse args
    splits = parse_splits(splits)  # type: ignore

    # Init config
    datasets = TimedGenerator(
        DatasetGenerator(
            use_cache=use_cache,
            return_interaction=True,
            genetic_split=splits[0] if splits and len(splits) == 1 else None,
            target=target_cat,
        )
    )
    model = datasets.generator.model  # type: ignore

    assert splits is not None

    assert (
        start_i < end_i
    ), f"Start index must be strictly less than end index: {start_i} < {end_i}"

    for i in range(start_i):
        printd(f"Skipping i = {i}")
        datasets.skip()

    for i in tqdm(range(start_i, end_i), disable=ConfigParams.DEBUG == 0):

        new_splits = []
        for split in splits:
            if not skip_sequence(i, target_cat, prev_df, split):
                new_splits.append(split)
            else:
                if pbar:
                    pbar.total -= 1
                    pbar.refresh()
                if target_cat:
                    printd(
                        f"Skipping i: {i} with target {target_cat} and split {split} because already in the log..."
                    )
                else:
                    printd(
                        f"Skipping i: {i} with split {split} because already in the log..."
                    )
        if len(new_splits) == 0:
            datasets.skip()
            continue

        splits = new_splits
        assert datasets.index == i, f"{datasets.index} != {i}"
        assert len(datasets.get_times()) == i, f"{len(datasets.get_times())} != {i}"

        try:
            dataset, interaction = next(datasets)
        except EmptyDatasetError as e:
            printd(f"run_full: Raised {type(e)}")
            logs = []
            for split in splits:
                alignment_log = log_alignment_error(
                    i=i,
                    error=error_messages[type(e)],
                    ks=ks,
                    split=split,
                    target_cat=target_cat,
                )
                logs.append(alignment_log)
            datasets.skip()
            datasets.generator.match_indices()  # type: ignore
            yield logs
            if pbar:
                pbar.update(1)
            continue
        except StopIteration:
            printd(f"STOP ITERATION RAISED")
            return

        assert (
            len(datasets.get_times()) == i + 1
        ), f"{len(datasets.get_times())} != {i+1}"

        source_sequence = interaction_to_tensor(interaction)  # type: ignore

        genetic_log = evaluate_genetic(
            i=i,
            target_cat=target_cat,
            datasets=datasets,
            dataset=dataset,
            source=source_sequence,
            model=model,
            ks=ks,
        )

        alignment_logs = []
        for split in splits:
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"| Split: {split}")
            alignment_log = evaluate_alignment(
                i=i,
                dataset=dataset,
                target_cat=target_cat,
                source=source_sequence,
                model=model,
                ks=ks,
                split=split,  # type: ignore
            )

            alignment_logs.append(alignment_log)

        logs = []
        for alignment_log in alignment_logs:
            logs.append({**genetic_log, **alignment_log})

        yield logs
