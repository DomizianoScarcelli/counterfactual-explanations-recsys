import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import fire
import os


def main(file_path: str, title: str,save_path: Optional[str] = None):
    if not save_path:
        file_name = os.path.basename(file_path).split(".")[0]+".png"
        save_path = f"plot/figs/{file_name}"
    if os.path.exists(save_path):
        os.remove(save_path)
    df = pd.read_csv(file_path)

    fidelity_columns = [
        "fidelity_score@1",
        "fidelity_score@5",
        "fidelity_score@10",
        "fidelity_score@20",
        "fidelity_gen_score@1",
        "fidelity_gen_score@5",
        "fidelity_gen_score@10",
        "fidelity_gen_score@20",
    ]
    df[fidelity_columns] = df[fidelity_columns] * 100  # Convert to percentages

    models = df["model"].unique()
    item_ids = df["gen_target_y@1"].astype(str).unique()
    methods = ["GENE", "PACE"]
    fidelity_metrics = [
        "fidelity_score@1",
        "fidelity_score@5",
        "fidelity_score@10",
        "fidelity_score@20",
    ]

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2  # Bar width
    colors = ["blue", "green", "orange", "red"]

    x_labels = []
    x_positions = []
    index = 0  # Start from 0, removing any offset

    # Assign unique positions dynamically
    for item_id in item_ids:
        for model in models:
            for method in methods:
                subset = df[
                    (df["gen_target_y@1"].astype(str) == item_id)
                    & (df["model"] == model)
                ]
                if not subset.empty:
                    x_positions.append(index)  # Store position for bar alignment
                    x_labels.append(f"{item_id}\n{model}\n{method}")
                    index += 1  # Move to the next position without gaps

    x_positions = np.array(x_positions)

    # Plot bars
    for i, metric in enumerate(fidelity_metrics):
        values = []
        for item_id in item_ids:
            for model in models:
                for method in methods:
                    subset = df[
                        (df["gen_target_y@1"].astype(str) == item_id)
                        & (df["model"] == model)
                    ]
                    if not subset.empty:
                        if method == "GENE":
                            value = subset[metric].values[0]
                        else:  # method == "PACE"
                            value = subset[
                                f"fidelity_gen_score@{str(metric.split('@')[-1])}"
                            ].values[0]
                    else:
                        value = 0  # Handle missing values
                    values.append(value)

        # Ensure no empty gaps at start
        ax.bar(
            x_positions - (width * 4) + (i * width),
            values,
            width,
            label=metric.replace("_score", ""),
            color=colors[i],
        )

    # Fix x-ticks alignment
    ax.set_xticks(x_positions - width * 0.5)  # Shift ticks to center
    ax.set_xticklabels(x_labels, rotation=0, ha="right", fontsize=9)

    # Labels and formatting
    ax.set_ylabel("Fidelity Score (%)")
    ax.set_title(title)
    ax.legend(title="")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    fire.Fire(main)
