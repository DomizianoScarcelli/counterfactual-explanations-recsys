from typing import List, Optional, Tuple

import pandas as pd
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

def compute_baseline_fidelity(df: pd.DataFrame) -> dict:
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

    similarity_threshold = pd.to_numeric(df["jaccard_threshold"], errors="coerce")

    # Iterate over each score column
    for score_col in score_columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        # Handle cases where values are None or error is not None
        if all(df["targeted"] == True):
            good_generation = (
                (df[score_col].notna()) & (df[score_col] > similarity_threshold)
            ).sum()
        elif all(df["targeted"] == False):
            good_generation = (
                (df[score_col].notna()) & (df[score_col] < similarity_threshold)
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
        if all(df["generation_strategy"] == "targeted") or all(
            df["generation_strategy"] == "targeted_uncategorized"
        ):
            good_generation = (
                (df[score_col].notna())
                & (df[error_col].isna())
                & (df[score_col] > similarity_threshold)
            ).sum()
        elif all(df["generation_strategy"] == "genetic") or all(
            df["generation_strategy"] == "genetic_categorized"
        ):
            good_generation = (
                (df[score_col].notna())
                & (df[error_col].isna())
                & (df[score_col] < similarity_threshold)
            ).sum()
        else:
            raise ValueError(
                f"The entire group should have the same generation strategy."
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
