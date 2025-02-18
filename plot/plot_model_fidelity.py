from utils import load_log
import os
from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np

def main(log_path: str):
    file_name = os.path.basename(log_path).split(".")[0]
    save_dir = "plot/figs/"
    os.makedirs(save_dir, exist_ok=True)
    
    df = load_log(log_path)

    fidelity_columns = [
        "fidelity_score_at_1",
        "fidelity_score_at_5",
        "fidelity_score_at_10",
        "fidelity_score_at_20",
        "fidelity_gen_score_at_1",
        "fidelity_gen_score_at_5",
        "fidelity_gen_score_at_10",
        "fidelity_gen_score_at_20",
    ]
    df[fidelity_columns] = df[fidelity_columns] * 100  # Convert to percentages

    models = df["model"].unique()
    item_ids = df["gen_target_y_at_1"].astype(str).unique()
    methods = ["GENE", "PACE"]
    fidelity_metrics = [
        "fidelity_score_at_1",
        "fidelity_score_at_5",
        "fidelity_score_at_10",
        "fidelity_score_at_20",
    ]
    generation_strategies = df["generation_strategy"].unique()

    for strategy in generation_strategies:
        strategy_df = df[df["generation_strategy"] == strategy]
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.2  # Bar width
        colors = ["blue", "green", "orange", "red"]

        x_labels = []
        x_positions = []
        index = 0  # Start from 0, removing any offset

        for item_id in item_ids:
            for model in models:
                for method in methods:
                    subset = strategy_df[
                        (strategy_df["gen_target_y_at_1"].astype(str) == item_id) & (strategy_df["model"] == model)
                    ]
                    if not subset.empty:
                        x_positions.append(index)
                        x_labels.append((item_id, model, method))
                        index += 1

        x_positions = np.array(x_positions)

        for i, metric in enumerate(fidelity_metrics):
            values = []
            valid_x_positions = []
            
            for idx, (item_id, model, method) in enumerate(x_labels):
                subset = strategy_df[
                    (strategy_df["gen_target_y_at_1"].astype(str) == item_id) & (strategy_df["model"] == model)
                ]
                if not subset.empty:
                    if method == "GENE":
                        value = subset[metric].values[0]
                    else:  # method == "PACE"
                        value = subset[f"fidelity_gen_score_at_{metric.split('_at_')[-1]}"]
                        value = value.values[0]
                else:
                    value = 0
                values.append(value)
                valid_x_positions.append(x_positions[idx])

            ax.bar(
                np.array(valid_x_positions) - (width * 4) + (i * width),
                values,
                width,
                label=metric.replace("_score", ""),
                color=colors[i],
            )

        ax.set_xticks(x_positions - width * 0.5)
        ax.set_xticklabels([f"{item}\n{model}\n{method}" for item, model, method in x_labels], 
                            rotation=0, ha="right", fontsize=9)

        ax.set_ylabel("Fidelity Score (%)")
        ax.set_title(f"Fidelity| {strategy}")
        ax.legend(title="")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        strategy_save_path = f"{save_dir}{file_name}_{strategy}.png"
        plt.savefig(strategy_save_path)

if __name__ == "__main__":
    fire.Fire(main)
