from typing import Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame


# Read the data from the item file


def frequencies(dataset: str, categorized: bool):
    if dataset in ["ml-1m", "ml-100k"]:
        return movielens_frequencies(dataset, categorized)
    elif dataset in ["steam"]:
        return steam_frequencies(categorized)


def steam_frequencies(categorized: bool) -> Tuple[DataFrame, DataFrame]:
    pass


def movielens_frequencies(
    dataset: str, categorized: bool
) -> Tuple[DataFrame, DataFrame]:
    item_df = pd.read_csv(
        f"dataset/{dataset}/{dataset}.item",
        sep="\t",
        names=["item_id", "movie_title", "release_year", "class"],
    )
    inter_df = pd.read_csv(
        f"dataset/{dataset}/{dataset}.inter",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    if categorized:
        df_exploded = item_df["class"].str.split(expand=True).stack()
    else:
        df_exploded = inter_df["item_id"].str.split(expand=True).stack()

    class_frequencies = df_exploded.value_counts()
    # if categorized:
    #     class_frequencies = class_frequencies.sort(on="item_id")
    # print("Class Frequencies:")
    # with pd.option_context(
    #     "display.max_rows", None, "display.max_columns", None, "display.width", None
    # ):
    # print(class_frequencies)
    if not categorized:
        return class_frequencies, None

    class_percentages = class_frequencies / len(item_df) * 100
    # print("\nClass Percentages:")
    # with pd.option_context(
    #     "display.max_rows", None, "display.max_columns", None, "display.width", None
    # ):
    #     print(class_percentages)

    merged_df = inter_df.merge(item_df[["item_id", "class"]], on="item_id")
    merged_df_exploded = merged_df["class"].str.split(expand=True).stack()

    interaction_class_frequencies = merged_df_exploded.value_counts()
    # print("Interaction-Level Class Frequencies:")
    # print(interaction_class_frequencies)

    interaction_class_percentages = (
        interaction_class_frequencies / len(merged_df_exploded) * 100
    )
    # print("\nInteraction-Level Class Percentages:")
    # print(interaction_class_percentages)
    return class_frequencies, interaction_class_frequencies


def plot(freqs: DataFrame, title: str, max_x_values: Optional[int] = None):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=freqs.index, y=freqs.values, palette="viridis")
    plt.xlabel("Categories")
    plt.ylabel("Frequency")
    if max_x_values is not None:
        indices = np.linspace(
            0, len(freqs) - 1, min(max_x_values, len(freqs)), dtype=int
        )
        plt.xticks(
            ticks=indices,
            labels=[freqs.index[i] for i in indices],
            rotation=45,
            ha="right",
        )
    else:
        plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    print(f"Figure plotted")
    plt.savefig(f"reports/plot/figs/{title}.png", bbox_inches="tight")
    plt.close()


def main(dataset: str):
    print("Starting...")
    class_freqs, interaction_freqs = frequencies(dataset, categorized=False)
    print(f"Uncategorized finished | {dataset}")
    plot(class_freqs, f"Uncategorized Class Frequencies | {dataset}", max_x_values=20)
    class_freqs, interaction_freqs = frequencies(dataset, categorized=True)
    class_freqs = class_freqs[class_freqs.index != "class:token_seq"]
    interaction_freqs = interaction_freqs[interaction_freqs.index != "class:token_seq"]
    print(f"Categorized finished | {dataset}")
    plot(class_freqs, f"Categorized Class Frequencies | {dataset}")
    plot(
        interaction_freqs,
        f"Categorized Interaction-Level Class Frequencies | {dataset}",
    )


if __name__ == "__main__":
    fire.Fire(main)
