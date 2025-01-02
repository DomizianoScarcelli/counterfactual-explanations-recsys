import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pandas as pd
from pandas import DataFrame

from performance_evaluation.alignment.utils import log_run
from run import run_full, run_genetic


def evaluate_trace_disalignment(
    range_i: Tuple[int, Optional[int]],
    splits: Optional[List[int]],
    use_cache: bool,
    mode: Literal["full", "genetic"] = "full",
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
    log: DataFrame = DataFrame({})
    if save_path and save_path.exists():
        log = pd.read_csv(save_path)

    if mode == "full":
        run_logs = run_full(
            start_i=range_i[0], end_i=range_i[1], splits=splits, use_cache=use_cache
        )
    elif mode == "genetic":
        run_logs = run_genetic(
            start_i=range_i[0], end_i=range_i[1], use_cache=use_cache, split=splits[0] if splits else None
        )
    else:
        raise ValueError(f"Mode '{mode}' not supported")

    for run_log in run_logs:
        # TODO: you can make Run a SkippableGenerator, which skips when the
        # source sequence, split and config combination already exists in the
        # log
        if save_path:
            log = log_run(
                prev_df=log,
                log=run_log,
                save_path=save_path,
                primary_key=["source", "split"],
            )


# def main(
#        config_path: Optional[str]=None,
#        mode: str = "evaluate",
#        use_cache: bool = True,
#        range_i: Tuple[int, Optional[int]] = (0, None),
#        log_path: Optional[str] = None,
#        stats_save_path: Optional[str] = None,
#        splits: Optional[List[int]] = None,
#        stat_filter: Optional[Dict[str, Any]]=None):
#    """
#    Main entry point to run either the trace disalignment evaluation or log statistics generation.

#    Args:
#        config_path: Path to the configuration file. If None, defaults will be used.
#        mode: Mode of operation. Can be "evaluate" (to evaluate traces) or "stats" (to generate statistics).
#        use_cache: Whether to use cached results when evaluating traces.
#        range_i: Tuple of indices (start, end) defining the range of evaluation.
#        log_path: Path to the log file for statistics. Required if mode is "stats".
#        stats_save_path: Path to save the generated statistics in JSON format. If None, not saved.
#        splits: List of splits to evaluate. If None, evaluates all splits.
#        stat_filter: Filter for statistics generation.

#    Returns:
#        None
#    """
#    set_seed()

#    ConfigParams.reload(config_path)
#    ConfigParams.fix()

#    if mode == "evaluate":
#        evaluate_trace_disalignment(
#                range_i=range_i,
#                splits=splits,
#                use_cache=use_cache,
#                save_path=log_path)
#    elif mode == "stats":
#        if not log_path:
#            raise ValueError(f"Log path needed for stats")
#        if not os.path.exists(log_path):
#            raise FileNotFoundError(f"File {log_path} does not exists")

#        stats_metrics = ["status", "dataset_time", "align_time", "automata_learning_time"]
#        group_by = list(ConfigParams.configs_dict().keys()) + ["split"]
#        group_by.remove("timestamp")
#        #TODO: temp
#        group_by.remove("include_sink")
#        group_by.remove("mutation_params")
#        group_by.remove("generation_strategy")
#        group_by.remove("fitness_alpha")
#        stats = get_log_stats(log_path=log_path, save_path=stats_save_path, group_by=group_by, metrics=stats_metrics, filter=stat_filter)
#        print(json.dumps(stats, indent=2))
#        return stats

#    else:
#        raise ValueError(f"Mode {mode} not supported, choose between [evaluate, stats]")


# if __name__ == "__main__":
#    fire.Fire(main)
