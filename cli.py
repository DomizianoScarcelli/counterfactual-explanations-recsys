from performance_evaluation.automata_learning.evaluate_with_test_set import (
    compute_automata_metrics,
)
import warnings

from pandas.api.types import is_int64_dtype

from performance_evaluation.automata_learning.evaluate_with_test_set import (
    run_automata_learning_eval,
)
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
from performance_evaluation.alignment.utils import compute_fidelity, log_run
from run import run_alignment
from run import run_alignment as run_alignment
from run import run_all, run_genetic
from sensitivity.model_sensitivity import main as evaluate_sensitivity
from sensitivity.model_sensitivity import run_on_all_positions
from utils import SeedSetter

RunModes: TypeAlias = Literal["alignment", "genetic", "automata_learning", "all"]


class RunSwitcher:
    def __init__(
        self,
        target: Optional[str],
        range_i: Tuple[int, Optional[int]],
        splits: Optional[List[int]],
        use_cache: bool,
        mode: RunModes = "all",
        save_path: Optional[Path] = None,
    ):
        self.targeted = ConfigParams.TARGETED
        self.categorized = ConfigParams.CATEGORIZED
        self.target = target
        if not self.targeted and self.target:
            warnings.warn(
                f"'target_cat': {self.target} won't be considered since 'targeted' is False"
            )
        self.splits = splits
        self.use_cache = use_cache
        self.mode = mode
        self.save_path = save_path
        self.ks = ConfigParams.TOPK

        self.start_i, self.end_i = range_i
        if self.end_i is None:
            temp_int = InteractionGenerator()
            self.end_i = sum(1 for _ in temp_int)

        self.targets: Optional[List[str]] = None
        if self.targeted and self.categorized:
            if self.target:
                self.targets = [self.target]
            else:
                self.targets = (
                    ConfigParams.TARGET_CAT
                    if ConfigParams.TARGET_CAT
                    else [cat for cat in cat2id if cat != "unknown"]
                )

            assert self.targets
            self.pbar = tqdm(
                desc=f"[TARGETED] Evaluating {self.end_i-self.start_i} sequences on {len(self.targets)} targets and {len(self.splits) if self.splits is not None else 1} splits",
                total=len(self.targets)
                * (self.end_i - self.start_i)
                * (len(self.splits) if self.splits is not None else 1),
            )
            self.og_desc = self.pbar.desc

        elif self.targeted and not self.categorized:
            raise NotImplementedError("Targeted uncategorized not yet implemented")
        elif not self.targeted:
            self.pbar = tqdm(
                desc=f"[UNTARGETED] Evaluating {self.end_i-self.start_i} sequences on {len(self.splits) if self.splits is not None else 1} splits",
                total=(self.end_i - self.start_i)
                * (len(self.splits) if self.splits is not None else 1),
            )
            self.og_desc = self.pbar.desc

    def run(self):
        if self.targeted:
            assert self.targets, "If 'targeted' is true, 'targets' must not be None"
            assert isinstance(self.targets, list) and isinstance(self.targets[0], str)
            for target in self.targets:
                self._run_single(target)
        else:
            self._run_single()

    def _run_single(self, target: Optional[str] = None):
        if target:
            self.pbar.set_description_str(self.og_desc + f" | Target: {target}")

        log: DataFrame = DataFrame({})
        if self.save_path and self.save_path.exists():
            log = pd.read_csv(self.save_path, dtype=str)

        if self.mode == "alignment":
            run_generator = run_alignment(
                target_cat=target,
                start_i=self.start_i,
                end_i=self.end_i,
                splits=self.splits,  # type: ignore
                use_cache=self.use_cache,
                ks=self.ks,
                prev_df=log,
                pbar=self.pbar,
            )
        elif self.mode == "genetic":
            run_generator = run_genetic(
                target_cat=target,
                start_i=self.start_i,
                end_i=self.end_i,
                split=self.splits[0] if self.splits and len(self.splits) == 1 else None,  # type: ignore
                prev_df=log,
                pbar=self.pbar,
            )  # NOTE: splits can be used in the genetic only if just a single one is used, otherwise each split would require a different dataset generation
        elif self.mode == "all":
            run_generator = run_all(
                target_cat=target,
                start_i=self.start_i,
                end_i=self.end_i,
                splits=self.splits,  # type: ignore
                use_cache=self.use_cache,
                ks=self.ks,
                prev_df=log,
                pbar=self.pbar,
            )
        else:
            raise ValueError(f"Mode '{self.mode}' not supported")

        for runs in run_generator:
            if not isinstance(runs, list):
                runs = [runs]
            for run in runs:
                if self.save_path:
                    if run["i"] is None:
                        continue

                    log_run(
                        prev_df=None,
                        log=run,
                        save_path=self.save_path,
                        # MAJOR TODO: change in the __init__ according to the targeted/untargeted setting
                        primary_key=["i", "source", "split", "gen_target_y@1"],
                        mode="append",
                        columns=list(log.columns),
                    )
                else:
                    print(json.dumps(run, indent=2))


def _absolute_paths(*paths: Optional[str]) -> Tuple[Optional[Path], ...]:
    return tuple(Path(path) if path is not None else None for path in paths)


class CLIStats:
    def stats(
        self,
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

    def sensitivity(
        self,
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
        save_path, config_path, log_path = _absolute_paths(
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

        return evaluate_sensitivity(
            mode="stats",
            groupby=group_by,  # type: ignore
            orderby=order_by,
            stats_save_path=save_path,
            log_path=log_path,
            target=target,  # type: ignore
            metrics=metrics,
        )

    def automata_metrics(self, log_path: str, save_path: Optional[str] = None):
        config_keys = list(ConfigParams.configs_dict().keys())
        df = pd.read_csv(log_path)
        config_keys.remove("timestamp")

        group_rows = []
        grouped = df.groupby(config_keys)

        for config_values, group in grouped:
            fidelity_dict = compute_automata_metrics(group)
            if isinstance(config_values, tuple):
                config_dict = dict(zip(config_keys, config_values))
            else:
                config_dict = {config_keys[0]: config_values}
            for key, value in fidelity_dict.items():
                config_dict[f"{key}"] = value
                config_dict[f"count"] = group.shape[0]
            group_rows.append(config_dict)

        metrics_df = pd.DataFrame(group_rows)

        if save_path:
            metrics_df.to_csv(save_path, index=False)
        else:
            print(metrics_df)

        pass

    def fidelity(self, log_path: str, save_path: Optional[str] = None):
        config_keys = list(ConfigParams.configs_dict().keys())
        df = pd.read_csv(log_path)
        if "gen_target_y@1" in df.columns:
            config_keys.append("gen_target_y@1")
            # TODO: after the target_cat=None but gen_target_y@1=target_cat encoded, adjust this accordingly
            config_keys.remove("target_cat")

        config_keys.remove("timestamp")
        config_keys.append("split")

        fidelity_rows = []
        grouped = df.groupby(config_keys)

        for config_values, group in grouped:
            fidelity_dict = compute_fidelity(group)
            if isinstance(config_values, tuple):
                config_dict = dict(zip(config_keys, config_values))
            else:
                config_dict = {config_keys[0]: config_values}
            for key, value in fidelity_dict.items():
                config_dict[f"fidelity_{key}"] = value
                config_dict[f"count"] = group.shape[0]
            fidelity_rows.append(config_dict)

        fidelity_df = pd.DataFrame(fidelity_rows)

        if save_path:
            fidelity_df.to_csv(save_path, index=False)
        else:
            print(fidelity_df)

    def alignment(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[ConfigDict] = None,
        log_path: Optional[str] = None,
        group_by: Optional[List[str] | str] = None,
        filter: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ):
        save_path, config_path, log_path = _absolute_paths(
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


class CLIEvaluate:

    def alignment(
        self,
        mode: RunModes = "all",
        target_cat: Optional[str] = None,
        range_i: Tuple[int, Optional[int]] = (0, None),
        splits: Optional[List[int]] = None,
        use_cache: bool = True,
        save_path: Optional[str] = None,
        config_path: Optional[str] = None,
        config_dict: Optional[ConfigDict] = None,
    ):
        save_path, config_path = _absolute_paths(save_path, config_path)
        if config_path and config_dict:
            raise ValueError(
                "Only one between config_path and config_dict must be set, not both"
            )
        if config_path:
            ConfigParams.reload(config_path)
        if config_dict:
            ConfigParams.override_params(config_dict)
        ConfigParams.fix()

        RunSwitcher(
            range_i=range_i,
            splits=splits,
            use_cache=use_cache,
            save_path=save_path,
            mode=mode,
            target=target_cat,
        ).run()

    def automata_learning(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[ConfigDict] = None,
        save_path: Optional[str] = None,
        end_i: int = 30,
    ):
        save_path, config_path = _absolute_paths(save_path, config_path)
        if config_path and config_dict:
            raise ValueError(
                "Only one between config_path and config_dict must be set, not both"
            )
        if config_path:
            ConfigParams.reload(config_path)
        if config_dict:
            ConfigParams.override_params(config_dict)
        ConfigParams.fix()

        run_automata_learning_eval(
            use_cache=False,
            end_i=end_i,
            save_path=save_path,
        )

    def sensitivity(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[ConfigDict] = None,
        save_path: Optional[str] = None,
        label_type: Optional[Literal["item", "category", "target"]] = None,
        k: Optional[int] = None,
    ):
        save_path, config_path = _absolute_paths(save_path, config_path)
        if config_path and config_dict:
            raise ValueError(
                "Only one between config_path and config_dict must be set, not both"
            )
        if config_path:
            ConfigParams.reload(config_path)
        if config_dict:
            ConfigParams.override_params(config_dict)
        ConfigParams.fix()

        if not k or not label_type:
            raise ValueError("k and target must not be None")

        return run_on_all_positions(label_type=label_type, log_path=save_path, k=k)


class CLIUtils:
    pass


class CLI:
    evaluate = CLIEvaluate()
    stats = CLIStats()
    utils = CLIUtils()

    def __init__(self):
        SeedSetter.set_seed()


if __name__ == "__main__":
    fire.Fire(CLI)
