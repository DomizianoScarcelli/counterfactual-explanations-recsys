import ast
from typing import List, Optional, Tuple

import fire
import pandas as pd
from pandas.core.api import DataFrame, Series

from constants import MAX_LENGTH


def calculate_metrics(
    row, df_stats: DataFrame, metrics: List[str], on: Tuple[str, str]
):
    split = ast.literal_eval(row["split"])
    s2 = split[1]
    s3 = split[2]

    min_pos_range, max_pos_range = (MAX_LENGTH - (s3+s2), MAX_LENGTH - (s3))
    # print(split, min_pos_range, max_pos_range)

    mask = (
        (df_stats[on[1]] == row[on[0]])
        # & (df_stats["position"] in pos_range)
        & (df_stats["position"] > min_pos_range)
        & (df_stats["position"] <= max_pos_range)
    )
    filtered_df1 = df_stats[mask]

    values = [
        filtered_df1[metric].mean() if not filtered_df1.empty else None
        for metric in metrics
    ]
    return Series(values)


def main(
    df_run: str | DataFrame,
    df_stats: str | DataFrame,
    metrics: List[str],
    on: Tuple[str, str] = ("source", "sequence"),
    save_path: Optional[str] = None,
):
    if isinstance(df_run, str):
        df_run = pd.read_csv(df_run)
    if isinstance(df_stats, str):
        df_stats = pd.read_csv(df_stats)

    merged = df_run.copy()
    merged[metrics] = df_run.apply(
        calculate_metrics, axis=1, df_stats=df_stats, metrics=metrics, on=on
    )

    if save_path:
        merged.to_csv(save_path, index=False)
    print(merged)
    return merged


if __name__ == "__main__":
    fire.Fire(main)
