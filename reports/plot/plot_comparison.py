import pandas as pd
import matplotlib.pyplot as plt


def clean_data(df):
    whitelist = {
        "ml-100k": [358, 657, 785, 1387],
        "steam": [333, 2, 1223, 448],
        "ml-1m": [105, 718, 248, 1761],
    }

    # Define a function that returns the correct whitelist for a given row
    def is_in_whitelist(row):
        if row["generation_strategy"] != "targeted_uncategorized":
            return True
        dataset = row["dataset"]
        target = int(row["target"])
        return target in whitelist.get(dataset, [])

    # Apply the filtering logic row by row
    df = df[df.apply(is_in_whitelist, axis=1)]
    return df


# Load the dataset
data = pd.read_csv("fidelity.csv")
baselines = pd.read_csv("baselines.csv")
baselines2edits = pd.read_csv("baselines_2edits.csv")

data["gen_target_y_at_1"] = data["gen_target_y_at_1"].apply(
    lambda x: (
        x.replace("{", "").replace("}", "")
        if isinstance(x, str) and x.startswith("{")
        else x
    )
)

# Define k values for fidelity
ks = [1, 5, 10, 20]
data = data.rename(columns={"gen_target_y_at_1": "target"})
for k in ks:
    data = data.rename(columns={f"fidelity_gen_score_at_{k}": f"fidelity@{k}"})
data["target"] = data["target"].fillna("None")

data = clean_data(data)

data = data[
    [
        "generation_strategy",
        "dataset",
        "model",
        *[f"fidelity@{k}" for k in ks],
        "target",
        "seed",
    ]
]

def plot_group(df, dataset, strategy, title: str):
    plt.figure(figsize=(8, 5))

    # Define k values and their corresponding column names
    k_values = [1, 5, 10, 20]
    fidelity_columns = [f"fidelity@{k}" for k in k_values]

    # Compute seed-level average first (average fidelity scores per target-model)
    seed_avg = df.groupby(["target", "model"])[fidelity_columns].mean().reset_index()

    # Compute model-level average (average fidelity scores across all targets)
    model_avg = seed_avg.groupby("model")[fidelity_columns].mean()

    # Plot fidelity scores across ks for each model
    for model in model_avg.index:
        plt.plot(k_values, model_avg.loc[model], marker="o", label=model)

    plt.xlabel("k (Fidelity@k)")
    plt.ylabel("Fidelity Score")
    plt.title(f"Fidelity Scores for {dataset} - {strategy}")
    plt.legend(title="Model")
    plt.xticks(k_values)
    plt.grid(True)

    # Save the plot
    plt.savefig(title, bbox_inches="tight")
    plt.close()


# Group by generation_strategy and dataset and plot
for (strategy, dataset), group in data.groupby(["generation_strategy", "dataset"]):
    plot_group(
        group, dataset, strategy, title=f"reports/plot/figs/{dataset}_{strategy}.png"
    )
