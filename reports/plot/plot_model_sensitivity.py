from typing import List

import fire
import matplotlib.pyplot as plt

from utils.utils import load_log


def get_title(targeted, categorized, dataset, target=None):
    title = "Sensitivity"
    title += " | Targeted" if targeted else " | Untargeted"
    title += "-Categorized" if categorized else "-Uncategorized"
    title += f" | {dataset}"
    if target not in [None, "None"]:  # Ensure target is valid
        title += f" | Target: {target}"
    return title


def main(log_path: str, whitelist_models: List[str]=[], whitelist_datasets: List[str]=[]):
    df = load_log(log_path)

    # Define score columns
    score_columns = ["score_at_1", "score_at_5", "score_at_10", "score_at_20"]

    df["target"] = df["target"].fillna("None")
    # Group by required columns and calculate mean
    group_by_columns = [
        "pos_from_end",
        "targeted",
        "categorized",
        "dataset",
        "model",
        "target",
    ]

    if whitelist_models:
        df = df[df["model"].isin(whitelist_models)]

    if whitelist_datasets:
        df = df[df["dataset"].isin(whitelist_datasets)]

    mean_scores = df.groupby(group_by_columns)[score_columns].mean().reset_index()

    # Get unique values for (targeted, categorized), dataset, and target (if applicable)
    strategies = mean_scores[["targeted", "categorized"]].drop_duplicates().values
    datasets = mean_scores["dataset"].unique()
    targets = mean_scores["target"].dropna().unique().tolist()

    # Ensure untargeted cases (where 'target' is missing or None) are included
    if not targets or mean_scores["target"].isna().any():
        targets.append(None)

    # Generate separate plots
    for targeted, categorized in strategies:
        print(f"[DEBUG] targeted: {targeted}, catgorized: {categorized}")
        for dataset in datasets:
            for target in targets:
                subset = mean_scores[
                    (mean_scores["targeted"] == targeted)
                    & (mean_scores["categorized"] == categorized)
                    & (mean_scores["dataset"] == dataset)
                ]

                # Properly handle untargeted cases
                if target not in [None, "None"]:
                    subset = subset[subset["target"] == target]
                else:
                    subset = subset[subset["target"] == "None"]  # Include NaN values

                # Skip if the subset is empty
                if subset.empty:
                    continue

                plt.figure(figsize=(10, 6))

                for model in subset["model"].unique():
                    model_subset = subset[subset["model"] == model]

                    for score in score_columns:
                        plt.plot(
                            model_subset["pos_from_end"],
                            model_subset[score],
                            marker="o",
                            label=f"{model} - {score.replace('score_at_', 'score@')}",
                        )

                plt.xlabel("Position from End")
                plt.ylabel("Mean Score")
                # plt.ylim(0, 1)
                plt.title(get_title(targeted, categorized, dataset, target))
                plt.legend()
                plt.grid()

                # Save each figure with a unique name
                target_suffix = (
                    f"_target{target}"
                    if target not in [None, "None"]
                    else "_untargeted"
                )
                plt.savefig(
                    f"plot/figs/sensitivity/sensitivity_targeted{targeted}_categorized{categorized}_dataset{dataset}{target_suffix}.png"
                )
                plt.close()


if __name__ == "__main__":
    fire.Fire(main)
