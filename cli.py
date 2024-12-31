from models.config_utils import generate_model
from utils_classes.generators import SequenceGenerator
from models.config_utils import get_config
from config import ConfigDict
from utils import SeedSetter
import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import fire

from config import ConfigParams
from performance_evaluation import alignment
from run import run_full as og_run
from sensitivity.model_sensitivity import (
    main as evaluate_sensitivity,
    run_on_all_positions,
)
from type_hints import RecDataset, RecModel


class CLI:
    def __init__(self):
        SeedSetter.set_seed()

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
            print(json.dumps(stats, indent=2))
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
        dataset_type: RecDataset = ConfigParams.DATASET,
        model_type: RecModel = ConfigParams.MODEL,
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
        ConfigParams.reload(config_path)
        ConfigParams.fix()
        # trick because run is a generator
        for _ in og_run(dataset_type, model_type, start_i, end_i, splits, use_cache):
            pass

    def evaluate(
        self,
        what: Optional[
            Literal["alignment", "generation", "automata_learning", "sensitivity"]
        ] = None,
        mode: Literal["full", "genetic"] = "full",
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
            alignment.evaluate.evaluate_trace_disalignment(
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
