import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias

import fire
import pandas as pd
from tqdm import tqdm

from config.config import ConfigDict, ConfigParams
from config.constants import cat2id
from core.evaluation.alignment.utils import (compute_edit_distance,
                                             compute_fidelity,
                                             compute_running_times)
from core.evaluation.automata_learning.evaluate_with_test_set import (
    compute_automata_metrics, run_automata_learning_eval)
from core.evaluation.evaluation_utils import compute_metrics
from run import run_alignment as run_alignment
from run import run_all, run_genetic
from scripts.dataframes.merge_dfs import main as merge_dfs_script
from scripts.print_pth import print_pth as print_pth_script
from scripts.targets_popularity import main as targets_popularity_script
from core.sensitivity.model_sensitivity import run_on_all_positions
from utils.utils import SeedSetter, load_log
from utils.generators import InteractionGenerator
from utils.utils import RunLogger

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
        if self.targeted:
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
            assert isinstance(self.targets, list) and isinstance(
                self.targets[0], (str, int)
            )
            for target in self.targets:
                self._run_single(target)
        else:
            self._run_single()

    def _run_single(self, target: Optional[str] = None):
        if target:
            self.pbar.set_description_str(self.og_desc + f" | Target: {target}")

        logger = RunLogger(
            db_path=self.save_path, schema=None, add_config=True, merge_cols=True
        )
        if self.mode == "alignment":

            run_alignment(
                target_cat=target,
                logger=logger,
                start_i=self.start_i,
                end_i=self.end_i,
                splits=self.splits,  # type: ignore
                use_cache=self.use_cache,
                ks=self.ks,
                pbar=self.pbar,
            )
        elif self.mode == "genetic":
            run_genetic(
                target_cat=target,
                logger=logger,
                start_i=self.start_i,
                end_i=self.end_i,
                split=self.splits[0] if self.splits and len(self.splits) == 1 else None,  # type: ignore
                pbar=self.pbar,
            )  # NOTE: splits can be used in the genetic only if just a single one is used, otherwise each split would require a different dataset generation
        elif self.mode == "all":
            run_all(
                target_cat=target,
                logger=logger,
                start_i=self.start_i,
                end_i=self.end_i,
                splits=self.splits,  # type: ignore
                use_cache=self.use_cache,
                ks=self.ks,
                pbar=self.pbar,
            )
        else:
            raise ValueError(f"Mode '{self.mode}' not supported")


def _absolute_paths(*paths: Optional[str]) -> Tuple[Optional[Path], ...]:
    return tuple(Path(path) if path is not None else None for path in paths)


class CLIScripts:
    def print_pth(self, pth_path: str):
        print_pth_script(pth_path)

    def targets_popularity(self, dataset: str):
        targets_popularity_script(dataset)

    def merge_dfs(
        self,
        paths: Optional[List[str]] = None,
        dir: Optional[str] = None,
        regex: Optional[str] = None,
        primary_key: List[str] = [],
        blacklist_keys: List[str] = ["timestamp"],
        ignore_config: bool = False,
        save_path: Optional[str] = None,
    ):
        merge_dfs_script(
            paths=paths,
            dir=dir,
            regex=regex,
            primary_key=primary_key,
            blacklist_keys=blacklist_keys,
            ignore_config=ignore_config,
            save_path=save_path,
        )


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

    def automata_metrics(self, log_path: str, save_path: Optional[str] = None):
        config_keys = list(ConfigParams.configs_dict().keys())
        df = load_log(log_path)

        print("[DEBUG], dataframe", df.info())

        columns_to_replace = ["tp", "tn", "fp", "fn", "precision", "accuracy", "recall"]
        df[columns_to_replace] = df[columns_to_replace].replace("None", 0.0)
        df[columns_to_replace] = df[columns_to_replace].fillna(0.0)
        for col in columns_to_replace:
            df[col] = df[col].astype(float)

        print("[DEBUG], dataframe", df.info())

        config_keys.remove("timestamp")

        group_rows = []
        grouped = df.groupby(config_keys)

        for config_values, group in grouped:
            metrics_dict = compute_automata_metrics(group)
            user_metrics_dict = compute_automata_metrics(group, average_per_user=True)

            if isinstance(config_values, tuple):
                config_dict = dict(zip(config_keys, config_values))
            else:
                config_dict = {config_keys[0]: config_values}

            for key, value in metrics_dict.items():
                config_dict[f"{key}"] = value
                config_dict[f"count"] = group.shape[0]

            for key, value in user_metrics_dict.items():
                config_dict[f"usr_avg_{key}"] = value
                config_dict[f"count"] = group.shape[0]

            group_rows.append(config_dict)

        metrics_df = pd.DataFrame(group_rows)

        # Compute "Average" rows for each category type
        for category_type in ["targeted", "targeted_uncategorized"]:
            category_df = metrics_df[metrics_df["generation_strategy"] == category_type]

            if not category_df.empty:
                avg_row = {"target_cat": "Average", "generation_strategy": category_type}

                # Sum tp, tn, fp, fn
                tp_sum = category_df["tp"].sum()
                tn_sum = category_df["tn"].sum()
                fp_sum = category_df["fp"].sum()
                fn_sum = category_df["fn"].sum()

                avg_row["tp"] = tp_sum
                avg_row["tn"] = tn_sum
                avg_row["fp"] = fp_sum
                avg_row["fn"] = fn_sum

                # Compute precision, accuracy, recall using summed values
                avg_row["precision"], avg_row["accuracy"], avg_row["recall"] = (
                    compute_metrics(tp=tp_sum, fp=fp_sum, tn=tn_sum, fn=fn_sum)
                )

                # Compute average user-level metrics
                avg_row["usr_avg_precision"] = category_df["usr_avg_precision"].mean()
                avg_row["usr_avg_accuracy"] = category_df["usr_avg_accuracy"].mean()
                avg_row["usr_avg_recall"] = category_df["usr_avg_recall"].mean()

                avg_row["count"] = category_df["count"].sum()

                metrics_df = pd.concat(
                    [metrics_df, pd.DataFrame([avg_row])], ignore_index=True
                )

        if save_path:
            metrics_df.to_csv(save_path, index=False)
        else:
            print(metrics_df)

    def fidelity(
        self, log_path: str, save_path: Optional[str] = None, clean: bool = True
    ):
        config_keys = list(ConfigParams.configs_dict().keys())
        df = load_log(log_path)
        if "gen_target_y_at_1" in df.columns:
            config_keys.append("gen_target_y_at_1")
            # TODO: after the target_cat=None but gen_target_y_at_1=target_cat encoded, adjust this accordingly
            config_keys.remove("target_cat")

        config_keys.remove("timestamp")
        if "split" in df.columns:
            config_keys.append("split")

        assert (
            set(config_keys) - set(df.columns) == set()
        ), f"Some config keys: {set(config_keys) - set(df.columns)} do not exist in the loaded df"

        df[config_keys] = df[config_keys].fillna("NaN")
        rows = []
        grouped = df.groupby(config_keys)

        for config_values, group in grouped:
            fidelity_dict = compute_fidelity(group)
            edit_distance_dict = compute_edit_distance(group)
            running_time_dict = compute_running_times(group)
            if isinstance(config_values, tuple):
                config_dict = dict(zip(config_keys, config_values))
            else:
                config_dict = {config_keys[0]: config_values}
            for key, value in fidelity_dict.items():
                config_dict[f"count"] = group.shape[0]
                config_dict[f"fidelity_{key}"] = value
            for key, value in {**edit_distance_dict, **running_time_dict}.items():
                config_dict[key] = value
            rows.append(config_dict)

        result_df = pd.DataFrame(rows)

        if save_path:
            result_df.to_csv(save_path, index=False)
        else:
            print(result_df)

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
        save_path: str,
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

        return run_on_all_positions(log_path=save_path, ks=ConfigParams.TOPK)


class CLIUtils:
    pass


class CLI:
    evaluate = CLIEvaluate()
    stats = CLIStats()
    utils = CLIUtils()
    scripts = CLIScripts()

    def __init__(self):
        SeedSetter.set_seed()


if __name__ == "__main__":
    fire.Fire(CLI)
