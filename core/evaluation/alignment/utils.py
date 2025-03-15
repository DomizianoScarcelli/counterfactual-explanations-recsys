import json
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Interaction
from torch import Tensor

from core.generation.dataset.utils import interaction_to_tensor
from core.models.utils import trim
from utils.utils import printd


def preprocess_interaction(
    raw_interaction: Interaction, oracle: Optional[SequentialRecommender] = None
) -> List[Tensor] | Tuple[List[Tensor], int]:
    source_sequence = interaction_to_tensor(raw_interaction)
    if not oracle:
        return trim(source_sequence.squeeze(0)).tolist()
    try:
        source_gt = oracle(source_sequence).argmax(-1).item()
    except IndexError as e:
        printd("IndexError on sequence ", source_sequence)
        raise e
    source_sequence = trim(source_sequence.squeeze(0)).tolist()
    return source_sequence, source_gt


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
                                   If `None`, the results are not saved_models.

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
        - If `save_path` is provided, the results are saved_models as a JSON file.

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


def compute_fidelity(df: pd.DataFrame) -> dict:
    """
    Compute fidelity for each score_at_k and gen_score_at_k in the DataFrame, handling None values
    and error cases.

    Args:
        df (pd.DataFrame): Input DataFrame containing score_at_k, gen_score_at_k columns,
                           jaccard_threshold column, and optionally an error column.

    Returns:
        dict: Dictionary containing fidelity_at_k values for each k.
    """
    fidelity_results = {}

    # Identify score and gen_score columns
    score_columns = [col for col in df.columns if col.startswith("score")]
    gen_score_columns = [col for col in df.columns if col.startswith("gen_score")]
    all_score_columns = score_columns + gen_score_columns

    similarity_threshold = pd.to_numeric(df["jaccard_threshold"], errors="coerce")

    # Iterate over each score column
    error_col = "error" if "error" in df.columns else "gen_error"
    for score_col in all_score_columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        # Handle cases where values are None or error is not None
        if all(df["gen_strategy"] == "targeted") or all(
            df["gen_strategy"] == "targeted_uncategorized"
        ):
            good_generation = (
                (df[score_col].notna())
                & (df[error_col].isna())
                & (df[score_col] > similarity_threshold)
            ).sum()
        elif all(df["gen_strategy"] == "genetic") or all(
            df["gen_strategy"] == "genetic_categorized"
        ):
            good_generation = (
                (df[score_col].notna())
                & (df[error_col].isna())
                & (df[score_col] < similarity_threshold)
            ).sum()
        else:
            raise ValueError(
                f"The entire group should have the same generation strategy. Insert the generation strategy as a primary key"
            )
        # Compute fidelity as the proportion of valid rows above the threshold
        fidelity_k = good_generation / len(df)

        # Store the result in the dictionary
        fidelity_results[score_col] = fidelity_k

    return fidelity_results


def compute_edit_distance(df: pd.DataFrame) -> dict:
    threshold = df.iloc[0]["jaccard_threshold"]
    strategy = df.iloc[0]["generation_strategy"]

    ks = [1, 5, 10, 20]
    results = {}

    for k in ks:
        gen_score_col = f"gen_score_at_{k}"
        score_col = f"score_at_{k}"
        gen_cost_col = f"gen_cost"
        cost_col = f"cost"

        if "targeted" in strategy:
            valid_gen_rows = df[df[gen_score_col] > threshold]
        else:
            valid_gen_rows = df[df[gen_score_col] < threshold]

        results[f"avg_gen_cost_at_{k}"] = valid_gen_rows[gen_cost_col].mean()

        if "targeted" in strategy:
            valid_cost_rows = df[df[score_col] > threshold]
        else:
            valid_cost_rows = df[df[score_col] < threshold]

        results[f"avg_cost_at_{k}"] = valid_cost_rows[cost_col].mean()

    return results


def compute_running_times(df: pd.DataFrame) -> dict:
    time_columns = [col for col in df.columns if col.endswith("time")]
    for col in time_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    time_means = {col: df[col].mean() for col in time_columns}
    return time_means
