import json
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from config import ConfigParams
from generation.dataset.utils import interaction_to_tensor
from models.utils import trim


def preprocess_interaction(
    raw_interaction: Interaction, oracle: Optional[SequentialRecommender] = None
) -> List[Tensor] | Tuple[List[Tensor], int]:
    source_sequence = interaction_to_tensor(raw_interaction)
    if not oracle:
        return trim(source_sequence.squeeze(0)).tolist()
    try:
        source_gt = oracle(source_sequence).argmax(-1).item()
    except IndexError as e:
        print("IndexError on sequence ", source_sequence)
        raise e
    source_sequence = trim(source_sequence.squeeze(0)).tolist()
    return source_sequence, source_gt


def pk_exists(
    df: pd.DataFrame, primary_key: List[str], consider_config: bool = True
) -> bool:
    """
    Returns True if a record with the same primary key exists in the dataframe.

    Args:
        df: the pandas DataFrame on which to look for the record.
        primary_key: the list of strings that are the primary key.
        consider_config: True if the config keys have to be included in the primary key.

    Returns:
        True if a record with the same primary key exists, otherwise False.
    """
    if df.empty:
        return False

    if consider_config:
        config_keys = list(ConfigParams.configs_dict().keys())
        config_keys.remove("timestamp")
        primary_key = primary_key + config_keys

    df = df.copy()  # Avoid modifying the original DataFrame
    df[primary_key] = df[primary_key].astype(str)
    return df[primary_key].duplicated().any()


def log_run(
    prev_df: DataFrame,
    log: Dict,
    save_path: str,
    primary_key: List[str] = [],
    add_config: bool = True,
) -> DataFrame:
    """
    Log the values in the log dict, concatenating them to a previous log,
    adding the configuration parameters, and saving the log to the disk as a
    pandas DataFrame.

    Args:
        prev_df: a pandas.Dataframe containing the previous logs, on which to concatenate the current log.
        log: the dictionary of the type: {"value": key} that contains the values to log.
        add_config: True if the configs found in the config.toml file have to be added to the log. Defaults to True
        primary_key: the set of fields that are unique for each record. If a
            record with the same primary key values exists in `prev_df`, the new
            record in log won't be added.
        save_path: the path where to save the log.

    Returns:
        The updated pandas.Dataframe
    """

    # Create a dictionary with input parameters as columns
    data = {}
    length = 1
    for key, value in log.items():
        if isinstance(value, list):
            if length == 1:
                length = len(value)
            data[key] = [round(v, 3) if isinstance(v, float) else v for v in value]
        else:
            data[key] = [round(value, 3) if isinstance(value, float) else value]

    configs = ConfigParams.configs_dict(length=length)

    if add_config:
        data = {**data, **configs}
        primary_key += list(configs.keys())
        primary_key.remove("timestamp")

    new_df = pd.DataFrame(data)
    # Remove the fields in primary key that do not exist in prev_df, otherwise key error
    primary_key = [field for field in primary_key if field in prev_df.columns]
    # Check for duplicates based on primary_key
    if not prev_df.empty:
        # find records in `new_df` that are not already in `prev_df` based on primary_key
        combined = pd.merge(new_df, prev_df, on=primary_key, how="left", indicator=True)
        new_records = combined[combined["_merge"] == "left_only"]
        if not new_records.empty:
            new_df = new_records[
                [col for col in combined.columns if not col.endswith("_y")]
            ].copy()
            # Rename columns to remove `_x` suffix
            new_df.columns = [
                col[:-2] if col.endswith("_x") else col for col in new_df.columns
            ]
            new_df = new_df.drop(columns=["_merge"])
        else:
            return prev_df
    prev_df = pd.concat([prev_df, new_df], ignore_index=True)
    prev_df.to_csv(save_path, index=False)
    return prev_df


def metric_mean(df: pd.DataFrame, metric_name: str) -> Union[float, Dict[str, int]]:
    # Check if the column exists
    if metric_name not in df.columns:
        raise ValueError(f"Column '{metric_name}' not found in the DataFrame.")
    column = df[metric_name]

    # If dtype is numeric, calculate the mean
    if pd.api.types.is_numeric_dtype(column):
        return float(column.mean())

    # If dtype is string, calculate the count of each unique value
    elif pd.api.types.is_string_dtype(column):
        return column.value_counts().to_dict()

    else:
        raise TypeError(f"Unsupported dtype for column '{metric_name}': {column.dtype}")


def stats_to_df(stats: List[Dict[str, Any]]) -> DataFrame:
    data = {}
    for stat in stats:
        for key, value in stat.items():
            if key not in data:
                data[key] = [round(value, 3) if isinstance(value, float) else value]
            else:
                data[key].append(round(value, 3) if isinstance(value, float) else value)
    return DataFrame(data)


def get_log_stats(
    log_path: str,
    group_by: List[str],
    metrics: List[str],
    filter: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compute statistics from a log file and optionally save the results.

    Args:
        log_path (str): Path to the CSV file containing the log data.
        group_by (List[str]): List of column names to group the data by.
                              Only columns present in the log file are considered.
        metrics (List[str]): List of column names for which the mean is computed
                             within each group.
        filter (Optional[Dict[str, Any]]): A dictionary specifying filters to apply.
                                           Each key is a column name, and its value
                                           is the expected value for that column.
                                           Only rows matching the filter criteria
                                           are included in the analysis.
                                           Example: `{"determinism": "true"}`.
        save_path (Optional[str]): Path to save the resulting statistics as a JSON file.
                                   If `None`, the results are not saved.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the computed statistics.
                              Each dictionary includes:
                              - Fields from `group_by`
                              - The number of rows in the group (`rows`)
                              - The mean value for each metric in `metrics`

    Notes:
        - Columns specified in `group_by` and `metrics` that are not present in the
          log file are ignored.
        - The function filters the data based on the `filter` dictionary before grouping,
          ensuring only relevant rows are included in the computation.
        - If `save_path` is provided, the results are saved as a JSON file.

    Example:
        ```python
        stats = get_log_stats(
            log_path="logs.csv",
            group_by=["model", "dataset"],
            metrics=["accuracy", "loss"],
            filter={"split": "test"},
            save_path="stats.json"
        )
        print(stats)
        ```
    """
    df = pd.read_csv(log_path)
    # Filter `group_by` to include only columns present in the DataFrame
    group_by = [field for field in group_by if field in df.columns]

    results = []
    for fields, group in df.groupby(group_by):
        result = {field_name: str(fields[i]) for i, field_name in enumerate(group_by)}  # type: ignore
        if filter and not all(
            result[filter_key] == filter_value
            for filter_key, filter_value in filter.items()
        ):
            continue
        averages = {"rows": group.shape[0]}
        for metric in metrics:
            averages[metric] = metric_mean(group, metric)
        results.append({**result, **averages})

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


# if __name__ == "__main__":
#     print(
#         get_log_stats(
#             "results/different_splits_run.csv",
#             ["pop_size", "generations"],
#             ["status", "dataset_time", "align_time"],
#             "test",
#         )
#     )
