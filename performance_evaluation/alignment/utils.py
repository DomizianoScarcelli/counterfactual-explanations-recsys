import datetime
import json
import time
from statistics import mean
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pandas import DataFrame


def log_run(df: DataFrame,
            original: List[int], 
            alignment: Optional[List[str]], 
            splits_key: str,
            genetic_key: Tuple[int, int, float], 
            status: str, 
            cost: int, 
            time_dataset_generation: float,
            time_automata_learning: float, 
            time_alignment: float,
            use_cache: bool) -> DataFrame:
    
    # Create a dictionary with input parameters as columns
    pop_size, generations, halloffame_ratio = genetic_key
    data = {
        "original_trace": [original],
        "alignment": [alignment],
        "splits_key": [splits_key],
        "population_size": [pop_size],
        "num_generations": [generations],
        "halloffame_ratio": [halloffame_ratio],
        "status": [status],
        "cost": [cost],
        "time_dataset_generation": [time_dataset_generation],
        "time_automata_learning": [time_automata_learning],
        "time_alignment": [time_alignment],
        "use_cache": [use_cache]
    }
    
    # Create a DataFrame from the dictionary
    new_df = pd.DataFrame(data)
    
    # Optionally append this new row to the existing DataFrame
    df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv("run.csv")
    
    return df


def evaluate_stats(log_path: str, stats_output: str) -> pd.DataFrame:
    """
    Given a log path, it calculates and returns statistics from the evaluation log
    as a DataFrame, with each row representing a unique combination of genetic parameters
    (population_size, generations, halloffame_ratio) and splits_key.

    Args:
        log_path: The path of the log CSV to load.
        stats_output: The path to save the output statistics CSV.
    
    Returns:
        A pandas DataFrame containing evaluation statistics.
    """
    # Load the DataFrame from the CSV log
    df = pd.read_csv(log_path)

    # Initialize a list to store results for DataFrame creation
    results = []

    # Group by 'splits_key', 'population_size', 'num_generations', and 'halloffame_ratio' to get stats for each group
    for (split_key, pop_size, generations, halloffame_ratio), group in df.groupby(
            ["splits_key", "population_size", "num_generations", "halloffame_ratio"]):
        
        stats = {
            "split_key": split_key,
            "population_size": pop_size,
            "num_generations": generations,
            "halloffame_ratio": halloffame_ratio,
            "total_runs": len(group),
            "good_runs": len(group[group["status"] == "good"]),
            "skipped_runs": len(group[group["status"] == "skipped"]),
            "bad_runs_counterfactual_not_found": len(group[group["status"] == "bad"]),
            "bad_runs_malformed_dfa": len(group[group["status"].isin(["DfaNotRejecting", "DfaNotAccepting"])]),
            "bad_runs_other": len(group[group["status"] == "skipped"]),
            "mean_time_dataset_generation": group.loc[group["status"] == "good", "time_dataset_generation"].mean(),
            "mean_time_automata_learning": group.loc[group["status"] == "good", "time_automata_learning"].mean(),
            "mean_time_alignment": group.loc[group["status"] == "good", "time_alignment"].mean(),
            "mean_total_time": (
                group.loc[group["status"] == "good", "time_dataset_generation"].mean() +
                group.loc[group["status"] == "good", "time_automata_learning"].mean() +
                group.loc[group["status"] == "good", "time_alignment"].mean()
            ),
            "min_cost": group.loc[group["status"] == "good", "cost"].min(),
            "max_cost": group.loc[group["status"] == "good", "cost"].max(),
            "mean_cost": group.loc[group["status"] == "good", "cost"].mean()
        }

        results.append(stats)

    # Create a DataFrame from the results list
    result_df = pd.DataFrame(results)

    # Save the result to a CSV file
    result_df.to_csv(stats_output, index=False)

    return result_df

