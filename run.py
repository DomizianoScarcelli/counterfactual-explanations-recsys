from exceptions import EmptyDatasetError
from constants import error_messages
from performance_evaluation.genetic.evaluate import (
    evaluate_targeted as evaluate_targeted_genetic,
)
from performance_evaluation.alignment.evaluate import (
    evaluate_targeted as evaluate_targeted_alignment,
)
from performance_evaluation.alignment.evaluate import log_error as log_alignment_error
from performance_evaluation.genetic.evaluate import log_error as log_genetic_error
import json
import warnings
from typing import Generator, List, Optional
from tqdm import tqdm

from config import ConfigParams
from generation.dataset.utils import interaction_to_tensor
from type_hints import SplitTuple
from utils_classes.generators import (
    DatasetGenerator,
    InteractionGenerator,
    TimedGenerator,
)
from utils_classes.Split import Split

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


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
    start_i: int = 0,
    end_i: Optional[int] = None,
    ks: List[int] = ConfigParams.TOPK,
    split: Optional[tuple] = None,  # type: ignore
):
    split: Split = parse_splits([split] if split else None)[0]  # type: ignore

    datasets = TimedGenerator(
        DatasetGenerator(
            use_cache=False,
            return_interaction=True,
            limit_generation_to="bad",
            genetic_split=split,
        )
    )
    model = datasets.generator.model  # type: ignore
    i = 0

    if end_i is None:
        temp_int = InteractionGenerator()
        end_i = sum(1 for _ in temp_int)
    for _ in range(start_i):
        datasets.skip()

    for i in range(start_i, end_i):
        try:
            dataset, interaction = next(datasets)
        except EmptyDatasetError as e:
            print(f"run_genetic: Raised {type(e)}")
            log = log_genetic_error(error=error_messages[type(e)], ks=ks)
            datasets.skip()
            datasets.generator.match_indices()  # type: ignore
            yield log
            continue
        except StopIteration:
            return

        # Obtain source categories
        source_sequence = interaction_to_tensor(interaction)  # type: ignore

        _, counterfactuals = dataset

        log = evaluate_targeted_genetic(
            i=i,
            datasets=datasets,
            source=source_sequence,
            counterfactuals=counterfactuals,
            model=model,
            ks=ks,
        )

        print(json.dumps(log, indent=2))
        yield log


def run_alignment(
    start_i: int = 0,
    end_i: Optional[int] = None,
    splits: Optional[List[tuple]] = None,  # type: ignore
    ks: List[int] = ConfigParams.TOPK,
    use_cache: bool = False,
) -> Generator:

    # Parse args
    splits = parse_splits(splits)  # type: ignore

    # Init config
    datasets = TimedGenerator(
        DatasetGenerator(
            use_cache=use_cache,
            return_interaction=True,
            genetic_split=splits[0],
        )
    )
    model = datasets.generator.model  # type: ignore

    assert splits is not None

    if end_i is None:
        temp_int = InteractionGenerator()
        end_i = sum(1 for _ in temp_int)
    assert (
        start_i < end_i
    ), f"Start index must be strictly less than end index: {start_i} < {end_i}"

    for i in range(start_i):
        print(f"Skipping i = {i}")
        datasets.skip()
        continue
    for i in tqdm(range(start_i, end_i)):
        assert datasets.index == i, f"{datasets.index} != {i}"
        assert len(datasets.get_times()) == i, f"{len(datasets.get_times())} != {i}"

        try:
            dataset, interaction = next(datasets)
        except StopIteration:
            print(f"STOP ITERATION RAISED")
            return
        except EmptyDatasetError as e:
            print(f"run_full: Raised {type(e)}")
            log = log_alignment_error(error=error_messages[type(e)], ks=ks)
            datasets.skip()
            datasets.generator.match_indices()  # type: ignore
            yield log
            continue

        assert (
            len(datasets.get_times()) == i + 1
        ), f"{len(datasets.get_times())} != {i+1}"

        source_sequence = interaction_to_tensor(interaction)  # type: ignore

        for split in splits:
            log = evaluate_targeted_alignment(
                i=i,
                dataset=dataset,
                source=source_sequence,
                model=model,
                ks=ks,
                split=split,  # type: ignore
            )

            print(json.dumps(log, indent=2))
            yield log
