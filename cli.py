from utils import printd
from utils_classes.generators import InteractionGenerator
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias

import fire
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from config import ConfigDict, ConfigParams
from constants import cat2id
from performance_evaluation import alignment
from performance_evaluation.alignment.utils import log_run
from run import run_alignment
from run import run_alignment as run_alignment
from run import run_all, run_genetic
from sensitivity.model_sensitivity import main as evaluate_sensitivity
from sensitivity.model_sensitivity import run_on_all_positions
from utils import SeedSetter

RunModes: TypeAlias = Literal["alignment", "genetic", "automata_learning", "all"]


def run_switcher(
    range_i: Tuple[int, Optional[int]],
    splits: Optional[List[int]],
    use_cache: bool,
    mode: RunModes = "all",
    save_path: Optional[Path] = None,
):
    """
    Evaluates the disalignment of traces for a given range and set of splits.
    Optionally saves the results to a CSV file.

    Args:
        range_i: The range of indices to evaluate (start, end).
        splits: List of split indices to evaluate. If None, evaluates all splits.
        use_cache: Whether to use cached results.
        save_path: Path to save the results to a CSV file. If None, results are not saved.

    Returns:
        None
    """
    targets: List[List[str]] = (
        ConfigParams.TARGET_CAT
        if ConfigParams.TARGET_CAT
        else [[cat] for cat in cat2id if cat != "unknown"]
    )  # type: ignore
    start_i, end_i = range_i
    if end_i is None:
        temp_int = InteractionGenerator()
        end_i = sum(1 for _ in temp_int)

    pbar = tqdm(
        desc=f"Evaluating {end_i-start_i} sequences on {len(targets)} targets and {len(splits) if splits is not None else 1} splits",
        total=len(targets)
        * (end_i - start_i)
        * (len(splits) if splits is not None else 1),
    )
    og_desc = pbar.desc

    for target in targets:
        pbar.set_description_str(og_desc + f" | Target: {target}")
        log: DataFrame = DataFrame({})
        if save_path and save_path.exists():
            log = pd.read_csv(save_path)

        if mode == "alignment":
            run_generator = run_alignment(
                target_cat=target,
                start_i=start_i,
                end_i=end_i,
                splits=splits,  # type: ignore
                use_cache=use_cache,
                ks=ConfigParams.TOPK,
                prev_df=log,
                pbar=pbar,
            )
        elif mode == "genetic":
            run_generator = run_genetic(
                target_cat=target,
                start_i=start_i,
                end_i=end_i,
                split=splits[0] if splits and len(splits) == 1 else None,  # type: ignore
                prev_df=log,
                pbar=pbar,
            )  # NOTE: splits can be used in the genetic only if just a single one is used, otherwise each split would require a different dataset generation
        elif mode == "all":
            run_generator = run_all(
                target_cat=target,
                start_i=start_i,
                end_i=end_i,
                splits=splits,  # type: ignore
                use_cache=use_cache,
                ks=ConfigParams.TOPK,
                prev_df=log,
                pbar=pbar,
            )
        else:
            raise ValueError(f"Mode '{mode}' not supported")

        for runs in run_generator:
            if not isinstance(runs, list):
                runs = [runs]
            for run in runs:
                printd(f"[DEBUG] Run is:", run)
                if save_path:
                    if run["i"] is None:
                        continue
                    log = log_run(
                        prev_df=log,
                        log=run,
                        save_path=save_path,
                        primary_key=["i", "source", "split", "gen_target_y@1"],
                    )
                else:
                    print(json.dumps(run, indent=2))


class CLI:
    def __init__(self):
        SeedSetter.set_seed()

    def _absolute_paths(self, *paths: Optional[str]) -> tuple:
        return tuple(Path(path) if path is not None else None for path in paths)

    def stats(
        self,
        what: Optional[
            Literal["alignment", "generation", "automata_learning", "sensitivity"]
        ] = None,
        config_path: Optional[str] = None,
        config_dict: Optional[ConfigDict] = None,
        log_path: Optional[str] = None,
        group_by: Optional[List[str] | str] = None,
        order_by: Optional[List[str] | str] = None,
        metrics: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        target: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Get statistics about previously run evaluations or analyze a given CSV file.

        Args:
            what : Type of evaluation to analyze. Defaults to None.
            config_path: Path to the configuration file. Defaults to None.
            log_path: Path to the log file for statistics. Required for "alignment". Defaults to None.
            group_by: Columns to group statistics by. Defaults to None.
            metrics: Metrics to include in the statistics. Defaults to None.
            filter: Filters to apply to the statistics. Defaults to None.
            save_path: Path to save the generated statistics in JSON format. Defaults to None.

        Returns:
            None or dict: Prints the statistics and optionally returns them as a dictionary.

        Raises:
            ValueError: If required parameters are missing or incorrect.
            FileNotFoundError: If the specified log file does not exist.

        Examples:
            1. Generate statistics for alignment:
                python -m cli stats alignment --log_path="path/to/log.csv"

            2. Generate sensitivity statistics grouped by sequence:
                python -m cli stats sensitivity --group_by=["sequence"]

            3. Generate general statistics for a CSV file:
                python -m cli stats --log_path="path/to/file.csv" --group_by=["column1"] --metrics=["metric1", "metric2"]
        """
        save_path, config_path, log_path = self._absolute_paths(
            save_path, config_path, log_path
        )

        if config_path and config_dict:
            raise ValueError(
                "Only one between config_path and config_dict must be set, not both"
            )
        if config_path:
            ConfigParams.reload(config_path)
        if config_dict:
            ConfigParams.override_params(config_dict)
        ConfigParams.fix()

        if what == "sensitivity":
            return evaluate_sensitivity(
                mode="stats",
                groupby=group_by,  # type: ignore
                orderby=order_by,
                stats_save_path=save_path,
                log_path=log_path,
                target=target,  # type: ignore
                metrics=metrics,
            )

        if what == "alignment":
            if not log_path:
                raise ValueError(f"Log path needed for stats")
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"File {log_path} does not exists")

            stats_metrics = [
                "status",
                "dataset_time",
                "align_time",
                "automata_learning_time",
            ]
            group_by = list(ConfigParams.configs_dict().keys()) + ["split"]
            group_by.remove("timestamp")

            stats = alignment.utils.get_log_stats(
                log_path=log_path,
                save_path=save_path,
                group_by=group_by,
                metrics=stats_metrics,
                filter=filter,
            )
            return stats

        if what == "generation":
            # TODO: implement
            raise NotImplementedError()

        if what == "automata_learning":
            # TODO: implement
            raise NotImplementedError()

        if not what and log_path and group_by and metrics:
            return alignment.utils.get_log_stats(
                log_path=log_path,
                group_by=group_by,
                metrics=metrics,
                filter=filter,
                save_path=save_path,
            )
        else:
            raise ValueError(
                "Error in parameters, run `python -m cli stats --help` for further information"
            )

    def run(
        self,
        config_path: Optional[str] = None,
        start_i: int = 0,
        end_i: Optional[int] = None,
        splits: Optional[List[tuple]] = None,  # type: ignore
        use_cache: bool = True,
    ):
        """
        Run counterfactual generations for a series of configurations.

        Args:
            config_path (Optional[str]): Path to the configuration file. Defaults to None.
            dataset_type (RecDataset): Type of dataset to use. Defaults to the value from `ConfigParams`.
            model_type (RecModel): Type of model to use. Defaults to the value from `ConfigParams`.
            start_i (int): Starting index of the evaluation range. Defaults to 0.
            end_i (Optional[int]): Ending index of the evaluation range. Defaults to None.
            splits (Optional[List[int]]): List of splits to evaluate. Defaults to None.
            use_cache (bool): Whether to use cached results. Defaults to True.

        Returns:
            None

        Examples:
            1. Run counterfactual generation with default configuration:
                python -m cli run

            2. Run counterfactual generation for a specific configuration file:
                python -m cli run --config_path="path/to/config.yaml"

            3. Run counterfactual generation for splits 0 and 1:
                python -m cli run --splits=[0,1]
        """
        config_path = self._absolute_paths(config_path)[0]

        ConfigParams.reload(config_path)
        ConfigParams.fix()
        # trick because run is a generator
        for _ in run_alignment(
            start_i=start_i,
            end_i=end_i,
            splits=splits,
            ks=ConfigParams.TOPK,
            use_cache=use_cache,
        ):
            pass

    def evaluate(
        self,
        what: Optional[
            Literal["alignment", "generation", "automata_learning", "sensitivity"]
        ] = None,
        mode: RunModes = "all",
        config_path: Optional[str] = None,
        config_dict: Optional[ConfigDict] = None,
        k: Optional[int] = None,
        label_type: Optional[Literal["item", "category", "target"]] = None,
        use_cache: bool = True,
        range_i: Tuple[int, Optional[int]] = (0, None),
        log_path: Optional[str] = None,
        splits: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ):
        """

        Evaluate trace disalignment or other specified tasks.

        Args:
            what (Optional[Literal["alignment", "generation", "automata_learning", "sensitivity"]]):
                Type of evaluation to perform. Defaults to None.
            config_path (Optional[str]): Path to the configuration file. Defaults to None.
            k (Optional[int]): Number of top results to consider for sensitivity analysis. Required for "sensitivity".
            target (Optional[Literal["item", "category"]]): Sensitivity analysis target type. Required for "sensitivity".
            use_cache (bool): Whether to use cached results. Defaults to True.
            range_i (Tuple[int, Optional[int]]): Range of indices (start, end) to evaluate. Defaults to (0, None).
            log_path (Optional[str]): Path to the log file for sensitivity evaluation. Required for "sensitivity".
            splits (Optional[List[int]]): List of splits to evaluate. Defaults to None.
            save_path (Optional[str]): Path to save the results. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If required parameters are missing or incorrect.
            NotImplementedError: If the specified evaluation type is not yet implemented.

        Examples:
            1. Evaluate trace disalignment:
                python -m cli evaluate alignment --range_i=(0, 100)

            2. Evaluate sensitivity analysis with `k=5` for items:
                python -m cli evaluate sensitivity --k=5 --target="item"

            3. Evaluate generation analysis (not implemented yet):
                python -m cli evaluate generation
        """
        save_path, config_path, log_path = self._absolute_paths(
            save_path, config_path, log_path
        )
        if config_path and config_dict:
            raise ValueError(
                "Only one between config_path and config_dict must be set, not both"
            )
        if config_path:
            ConfigParams.reload(config_path)
        if config_dict:
            ConfigParams.override_params(config_dict)
        ConfigParams.fix()

        if what == "alignment":
            run_switcher(
                range_i=range_i,
                splits=splits,
                use_cache=use_cache,
                save_path=save_path,
                mode=mode,
            )
        if what == "sensitivity":
            if not k or not label_type:
                raise ValueError("k and target must not be None")

            return run_on_all_positions(label_type=label_type, log_path=log_path, k=k)
            # both ends included
        if what == "generation":
            # TODO: implement
            raise NotImplementedError()
        if what == "automata_learning":
            # TODO: implement
            raise NotImplementedError()

    def utils(self):
        # TODO: insert csv utils that allow pipe-read and pipe-write to modify the csv files on the fly
        raise NotImplementedError()


if __name__ == "__main__":
    fire.Fire(CLI)
