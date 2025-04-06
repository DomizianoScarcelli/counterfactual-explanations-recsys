from utils.utils import load_log
import pandas as pd
from pandas import DataFrame

baseline = load_log("results/evaluate/baselines.db")
baseline["num_edits"] = 1
baseline_2edits = load_log("results/evaluate/baselines_2edits.db")
baseline_2edits["num_edits"] = 2


def combine_baselines(b1: DataFrame, b2: DataFrame) -> DataFrame:
    # Concatenate the two dataframes
    combined = pd.concat([b1, b2], ignore_index=True)

    # Define sort order (descending fidelity scores, ascending num_edits)
    sort_cols = [
        *[f"score_at_{k}" for k in [1, 5, 10, 20]],
        "num_edits",
    ]
    ascending = [False, False, False, False, True]

    # Sort and drop duplicates, keeping the best row per combination
    combined_sorted = combined.sort_values(by=sort_cols, ascending=ascending)
    subset = [
        "i",
        "generation_strategy",
        "dataset",
        "baseline",
        "model",
        "target",
        "num_edits",
        "seed",
    ]
    best_rows = combined_sorted.drop_duplicates(subset=subset, keep="first")

    return best_rows.reset_index(drop=True)


baselines = combine_baselines(baseline, baseline_2edits)
baselines.to_csv("results/evaluate/combined_baselines.csv")
