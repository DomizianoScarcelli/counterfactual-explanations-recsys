from pathlib import Path
import warnings
from typing import List, Optional, Set


from config.config import ConfigParams
from config.constants import cat2id, error_messages
from core.evaluation.genetic.evaluate import evaluate_genetic
from core.evaluation.genetic.evaluate import log_error as log_genetic_error
from core.generation.dataset.utils import interaction_to_tensor
from exceptions import EmptyDatasetError
from type_hints import SplitTuple
from utils.generators import DatasetGenerator, TimedGenerator
from utils.Split import Split
from utils.utils import RunLogger, printd

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def skip_sequence(
    i: int,
    primary_key: List[str],
    target_cat: Optional[str],
    logger: RunLogger,
    split: tuple,
):
    if logger.is_empty():
        return False
    new_row = {"i": i}
    if target_cat:
        new_row["gen_target_y@1"] = (
            f"{'{'}{cat2id()[target_cat]}{'}'}"
            if isinstance(target_cat, str)
            else f"{'{'}{target_cat}{'}'}"
        )
    new_row["split"] = str(split)

    new_row.update(ConfigParams.configs_dict(pandas=False, tostr=True))

    exists = logger.exists(log=new_row, primary_key=primary_key, consider_config=True)
    if exists:
        printd(f"Skipping in {logger.db_path}", new_row)
    return exists


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
    logger: RunLogger,
    start_i: int = 0,
    end_i: int = 1,
    sampled_indices: Optional[Set[int]] = None,
    ks: List[int] = ConfigParams.TOPK,
    split: Optional[tuple] = None,  # type: ignore
    pbar=None,
):
    split: Split = parse_splits([split] if split else None)[0]  # type: ignore
    primary_key = ["i", "generation_strategy", "split"]
    if target_cat:
        primary_key.append("gen_target_y@1")

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
        if sampled_indices and i not in sampled_indices:
            printd(f"Skipping i = {i} because it was not sampled")
            datasets.skip()
            continue
        if skip_sequence(i, primary_key, target_cat, logger, split.split):
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
        except (EmptyDatasetError, KeyError) as e:
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

            logger.log_run(log, primary_key=primary_key)

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
        log["split"] = str(split)

        logger.log_run(log, primary_key=primary_key)