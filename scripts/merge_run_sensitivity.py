from typing import List, Optional, Tuple
import pandas as pd
import fire
import ast

from pandas.core.api import DataFrame, Series


def calculate_metrics(
    row, df_stats: DataFrame, metrics: List[str], on: Tuple[str, str]
):
    split = ast.literal_eval(row["split"])
    s2 = split[1]
    s3 = split[2]

    mask = (
        (df_stats[on[0]] == row[on[1]])
        & (df_stats["position"] > s3)
        & (df_stats["position"] <= s2)
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
    print(f"Merged is:", merged)
    return merged


if __name__ == "__main__":
    fire.Fire(main)
