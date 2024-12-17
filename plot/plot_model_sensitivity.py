import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("results/stats/sensitivity/edit_model_sensitivity_cat_new_columns.csv")


# Filter relevant columns
x = ["position"]
# y = ['all_changes', 'any_changes', 'jaccards']
y = ["ndcg", "not counterfactual", "jaccards"]
columns_to_use = x + y
# Group data by 'k'
k_values = data["k"].unique() if "k" in data else None

if k_values is not None and len(k_values.tolist()) != 0:
    for k in k_values:
        # Filter data for the current k value
        subset = data[data["k"] == k][columns_to_use]

        # Plot metrics for different positions
        plt.figure(figsize=(10, 6))
        for metric in y:
            plt.plot(
                subset[x[0]],
                subset[metric],
                label=f"Mean {metric}",
                marker="o",
            )

        plt.title(f"Metrics for k = {k}")
        plt.xlabel("Changed element at index")
        plt.ylabel("Metrics")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.savefig(f"plot/figs/model_sensitivity_at_{k}.png")
else:
    subset = data[columns_to_use]
    # Plot metrics for different positions
    plt.figure(figsize=(10, 6))

    for metric in y:
        plt.plot(
            subset[x[0]],
            subset[metric],
            label=f"Mean {metric}",
            marker="o",
        )

    plt.title(f"Metrics for categorized (k = 1)")
    plt.xlabel("Changed element at index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"plot/figs/model_sensitivity_category.png")
