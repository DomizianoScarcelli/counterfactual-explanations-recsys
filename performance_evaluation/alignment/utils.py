from typing import List, Optional, Tuple, Dict

import pandas as pd
from pandas import DataFrame
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from config import GENERATIONS, HALLOFFAME_RATIO, POP_SIZE, DETERMINISM, MODEL, DATASET, ALLOWED_MUTATIONS
from genetic.dataset.utils import get_sequence_from_interaction
from models.utils import trim
from type_hints import SplitTuple


def get_split(slen: int, split_type: str) -> Tuple[str, SplitTuple]:
    mut_map = {f"{i}_mut": ((slen-i)/slen, i/slen, 0.0)  for i in range(1, slen)}
    nth_mut_map = {f"{i}th_mut": ((slen-i-1)/slen, 1/slen, i/slen)  for i in range(1, slen)}
    _map = {**mut_map, **nth_mut_map}
    if split_type not in _map:
        raise ValueError(f"Split type '{split_type}' not recognized")
    return split_type, _map[split_type]

def is_already_evaluated(log: DataFrame, sequence: List[int], splits_key: str) -> bool:
    """ Returns True if the sequenece has already been evaluted with the same evaluation parameters.

    Args:
        log: The pandas.Dataframe that contains the evaluated sequences with the relative evaluation
        sequence: The trimmed and flattened source sequence.

    Returns:
        True if the sequence is in the log with the same evaluation parameters, false oterwise.
    """
    return log.shape[0] != 0 and \
        log[(log["original_trace"].apply(lambda x: x == sequence)) & 
            (log["splits_key"] == splits_key) & 
            (log["population_size"] == POP_SIZE) &
            (log["num_generations"] == GENERATIONS) & 
            (log["halloffame_ratio"] == HALLOFFAME_RATIO)].shape[0] > 0

def preprocess_interaction(raw_interaction: Interaction, oracle: Optional[SequentialRecommender]=None) -> List[Tensor] | Tuple[List[Tensor], int]:
    source_sequence = get_sequence_from_interaction(raw_interaction)
    if not oracle:
        return trim(source_sequence.squeeze(0)).tolist()
    try:
        source_gt = oracle.full_sort_predict(source_sequence).argmax(-1).item()
    except IndexError as e:
        print("IndexError on sequence ", source_sequence)
        raise e
    source_sequence = trim(source_sequence.squeeze(0)).tolist()
    return source_sequence, source_gt

def log_run(prev_df: DataFrame,
            log: Dict,
            save_path: str) -> DataFrame:
    
    # Create a dictionary with input parameters as columns
    data = {key: [value] for key, value in log.items()}
    
    configs = {
            "determinism": [DETERMINISM],
            "model": [MODEL],
            "datset": [DATASET],
            "generations": [GENERATIONS],
            "halloffame_ratio": [HALLOFFAME_RATIO],
            "allowed_mutations": [ALLOWED_MUTATIONS]}

    data = {**data, **configs}

    new_df = pd.DataFrame(data)                               
    prev_df = pd.concat([prev_df, new_df], ignore_index=True).drop_duplicates()

    prev_df.to_csv(save_path, index=False)
    
    return prev_df


def evaluate_stats(log_path: str, stats_output: str) -> pd.DataFrame:
    #TODO: modify it to work on the new log
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

