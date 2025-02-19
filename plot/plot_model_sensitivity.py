import fire
import matplotlib.pyplot as plt

from utils import load_log


def get_title(targeted, categorized, dataset):
    title = "Sensitivity"
    if targeted:
        title += " | Targeted"
    else:
        title += " | Unargeted"
    if categorized:
        title += "-Categorized"
    else:
        title += "-Uncategorized"
    title += f" | {dataset}"
    return title


def main(log_path: str):
    df = load_log(log_path)

    # Define score columns
    score_columns = ["score_at_1", "score_at_5", "score_at_10", "score_at_20"]

    # Group by required columns and calculate mean
    mean_scores = (
        df.groupby(["pos_from_end", "targeted", "categorized", "dataset", "model"])[
            score_columns
        ]
        .mean()
        .reset_index()
    )

    # Get unique values for (targeted, categorized) and dataset
    strategies = mean_scores[["targeted", "categorized"]].drop_duplicates().values
    datasets = mean_scores["dataset"].unique()

    # Generate separate plots for each (targeted, categorized, dataset) combination
    for targeted, categorized in strategies:
        for dataset in datasets:
            plt.figure(figsize=(10, 6))

            subset = mean_scores[
                (mean_scores["targeted"] == targeted)
                & (mean_scores["categorized"] == categorized)
                & (mean_scores["dataset"] == dataset)
            ]

            for model in subset["model"].unique():
                model_subset = subset[subset["model"] == model]

                for score in score_columns:
                    plt.plot(
                        model_subset["pos_from_end"],
                        model_subset[score],
                        marker="o",
                        label=f"{model} - {score}",
                    )

            plt.xlabel("Position from End")
            plt.ylabel("Mean Score")
            plt.title(get_title(targeted, categorized, dataset))
            plt.legend()
            plt.grid()

            # Save each figure with a unique name
            plt.savefig(
                f"plot/figs/sensitivity_targeted{targeted}_categorized{categorized}_dataset{dataset}.png"
            )
            plt.close()


if __name__ == "__main__":
    fire.Fire(main)
