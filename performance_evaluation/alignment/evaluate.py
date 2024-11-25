import os
from typing import Optional, List, Tuple

import fire
import pandas as pd
from pandas import DataFrame
from config import ConfigParams
from performance_evaluation.alignment.utils import log_run, evaluate_stats
from run import run
from utils import set_seed


def evaluate_trace_disalignment(range_i: Tuple[int, Optional[int]],
                                splits: Optional[List[int]],
                                use_cache: bool,
                                save_path: str):
                                
    log: DataFrame  = DataFrame({})
    if os.path.exists(save_path):
        log = pd.read_csv(save_path)

    run_logs = run(start_i=range_i[0], 
                   end_i=range_i[1],
                   splits=splits,
                   use_cache=use_cache)

    for run_log in run_logs:
        # TODO: you can make Run a SkippableGenerator, which skips when the
        # source sequence, split and config combination already exists in the
        # log
        log = log_run(prev_df=log, 
                      log=run_log, 
                      save_path=save_path,
                      primary_key=["source", "split"])  

def main(
        config_path: Optional[str]=None,
        mode: str = "evaluate", 
        use_cache: bool = True, 
        range_i: Tuple[int, Optional[int]] = (0, None),
        evaluation_log: Optional[str] = None,
        stats_output: Optional[str] = None,
        splits: Optional[List[int]] = None,
        save_path: str = "results/evaluate.csv"):
    set_seed()

    ConfigParams.reload(config_path)
    ConfigParams.fix()

    if mode == "evaluate":
        evaluate_trace_disalignment(
                range_i=range_i,
                splits=splits, 
                use_cache=use_cache,
                save_path=save_path)
    elif mode == "stats":
        if not evaluation_log:
            raise ValueError("Evaluation log path needed for stats")
        if not os.path.exists(evaluation_log):
            raise FileNotFoundError(f"File {evaluation_log} does not exists")
        if not stats_output:
            raise ValueError("Evaluation stats output needed")
        evaluate_stats(evaluation_log, stats_output)

    else:
        raise ValueError(f"Mode {mode} not supported, choose between [evaluate, stats]")



if __name__ == "__main__":
    fire.Fire(main)
    
