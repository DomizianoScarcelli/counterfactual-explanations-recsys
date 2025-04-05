from scripts.preprocess_dataset import main as preprocess_dataset
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias
from core.generation.strategies.baselines import main as compute_baselines

import fire
import pandas as pd
from tqdm import tqdm

from bin.run import run_genetic
from config.config import ConfigDict, ConfigParams
from config.constants import cat2id
from core.evaluation.genetic.utils import (
    compute_baseline_fidelity,
    compute_edit_distance,
    compute_fidelity,
    compute_running_times,
)
from core.evaluation.evaluation_utils import compute_metrics
from core.sensitivity.model_sensitivity import run_on_all_positions
from scripts.print_pth import print_pth as print_pth_script
from scripts.targets_popularity import main as targets_popularity_script
from type_hints import RecDataset
from utils.generators import InteractionGenerator
from utils.utils import RunLogger, SeedSetter, load_log

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
            sample_num: Optional[int] = None,
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

        self.sampled_indices = None
        if sample_num:
            sample_range = range(self.start_i, self.end_i)
            if sample_num > len(sample_range):
                raise ValueError(
                    f"sample_num ({sample_num}) must be smaller than sample range ({len(sample_range)})"
                )
            self.sampled_indices = set(
                random.sample(population=sample_range, k=sample_num)
            )
        self.targets: Optional[List[str]] = None

        num_sequences = (
            len(self.sampled_indices)
            if self.sampled_indices
            else self.end_i - self.start_i
        )
        if self.targeted:
            if self.target:
                self.targets = [self.target]
            else:
                self.targets = (
                    [ConfigParams.TARGET_CAT]
                    if ConfigParams.TARGET_CAT
                    else [cat for cat in cat2id() if cat != "unknown"]
                )

            assert self.targets
            self.pbar = tqdm(
                desc=f"[TARGETED] Evaluating {num_sequences} sequences on {len(self.targets)} targets and {len(self.splits) if self.splits is not None else 1} splits",
                total=len(self.targets)
                      * num_sequences
                      * (len(self.splits) if self.splits is not None else 1),
            )
            self.og_desc = self.pbar.desc

        elif not self.targeted:
            self.pbar = tqdm(
                desc=f"[UNTARGETED] Evaluating {num_sequences} sequences on {len(self.splits) if self.splits is not None else 1} splits",
                total=num_sequences
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
        run_genetic(
            target_cat=target,
            logger=logger,
            start_i=self.start_i,
            end_i=self.end_i,
            sampled_indices=self.sampled_indices,
            split=self.splits[0] if self.splits and len(self.splits) == 1 else None,  # type: ignore
            pbar=self.pbar,
        )  # NOTE: splits can be used in the genetic only if just a single one is used, otherwise each split would require a different dataset generation


def _absolute_paths(*paths: Optional[str]) -> Tuple[Optional[Path], ...]:
    return tuple(Path(path) if path is not None else None for path in paths)


class CLIScripts:
    def print_pth(self, pth_path: str):
        print_pth_script(pth_path)

    def targets_popularity(self, dataset: str, save_plot: bool = False):
        targets_popularity_script(dataset, save_plot)

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
        """
    def fidelity(
            self, log_path: str, save_path: Optional[str] = None, clean: bool = True
    ):
        config_keys = list(ConfigParams.configs_dict().keys())
        df = load_log(log_path)
        if "gen_target_y_at_1" in df.columns:
            config_keys.append("gen_target_y_at_1")
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

    def baseline_fidelity(self, log_path: str, save_path: Optional[str] = None):
        config_keys = list(ConfigParams.configs_dict().keys())
        df = load_log(log_path)
        config_keys.append("target")
        config_keys.append("targeted")
        config_keys.append("categorized")
        config_keys.append("baseline")
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
            fidelity_dict = compute_baseline_fidelity(group)
            if isinstance(config_values, tuple):
                config_dict = dict(zip(config_keys, config_values))
            else:
                config_dict = {config_keys[0]: config_values}
            for key, value in fidelity_dict.items():
                config_dict[f"count"] = group.shape[0]
                config_dict[f"fidelity_{key}"] = value
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
            range_i: Tuple[int, Optional[int]] = (0, None),
            sample_num: Optional[int] = None,
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
            sample_num=sample_num,
            splits=splits,
            use_cache=use_cache,
            save_path=save_path,
            mode=mode,
            target=(
                ConfigParams.TARGET_CAT
                if ConfigParams.TARGET_CAT is not False
                else None
            ),
        ).run()

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
    def preprocess(self, dataset):
        for key in RecDataset:
            if key.value == dataset:
                recdataset = key.name
        config_dict: ConfigDict = {"settings": {"dataset": recdataset}}
        ConfigParams.override_params(config_dict)
        ConfigParams.fix()
        preprocess_dataset(dataset)

    def baselines(
            self,
            dataset,
            baseline,
            model="BERT4Rec",
            num_samples: int = 200,
            num_edits: int = 1,
            seed: Optional[int] = None,
    ):
        for key in RecDataset:
            if key.value == dataset:
                recdataset = key.name
        config_dict: ConfigDict = {
            "settings": {
                "dataset": recdataset,
                "seed": seed if seed else 42,
                "model": model,
            }
        }
        config_dict.update({"evolution": {"pop_size": 1024}})
        ConfigParams.override_params(config_dict)
        ConfigParams.fix()
        compute_baselines(
            baseline=baseline, num_samples=num_samples, num_edits=num_edits
        )


class CLI:
    evaluate = CLIEvaluate()
    stats = CLIStats()
    utils = CLIUtils()
    scripts = CLIScripts()

    def __init__(self):
        SeedSetter.set_seed()


if __name__ == "__main__":
    fire.Fire(CLI)
