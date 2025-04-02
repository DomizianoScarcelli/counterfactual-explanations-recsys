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


def preprocess_fidelity_df(df):
    df["gen_target_y_at_1"] = df["gen_target_y_at_1"].apply(
        lambda x: (
            x.replace("{", "").replace("}", "")
            if isinstance(x, str) and x.startswith("{")
            else x
        )
    )
    ks = [1, 5, 10, 20]
    df = df.rename(columns={"gen_target_y_at_1": "target"})
    for k in ks:
        df = df.rename(columns={f"fidelity_gen_score_at_{k}": f"fidelity@{k}"})

    df["target"] = df["target"].fillna("None")
    df = clean_data(df)
    df = df[
        [
            "generation_strategy",
            "dataset",
            "model",
            *[f"fidelity@{k}" for k in ks],
            "target",
            "seed",
        ]
    ]
    return df


def preprocess_baselines_df(df):
    baseline_mapping = {
        "ml-100k": {50: 358, 411: 657, 630: 785, 1305: 1387},
        "steam": {271590: 333, 35140: 2, 292140: 1223, 582160: 448},
        "ml-1m": {2858: 105, 2005: 718, 728: 248, 2738: 1761},
    }
    ks = [1, 5, 10, 20]

    # Rename fidelity columns
    for k in ks:
        df = df.rename(columns={f"fidelity_score_at_{k}": f"fidelity@{k}"})
    # Fill missing target values with "None"
    df["target"] = df["target"].fillna("None")

    # Map target values based on dataset
    def map_target(row):
        dataset = row["dataset"]
        target = int(row["target"])
        return baseline_mapping[dataset][target]

    df["target"] = df.apply(map_target, axis=1)

    # Implement the TODO logic to generate the correct generation_strategy
    def map_generation_strategy(row):
        if row["targeted"] and row["categorized"]:
            return "targeted"
        elif row["targeted"] and not row["categorized"]:
            return "targeted_uncategorized"
        elif not row["targeted"] and not row["categorized"]:
            return "genetic"
        elif not row["targeted"] and row["categorized"]:
            return "genetic_categorized"

    # Apply the mapping function to the DataFrame
    df["generation_strategy"] = df.apply(map_generation_strategy, axis=1)

    # Select relevant columns for the final DataFrame
    df = df[
        [
            "generation_strategy",
            "dataset",
            "baseline",
            "model",
            *[f"fidelity@{k}" for k in ks],
            "target",
            "seed",
        ]
    ]

    return df


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


# Load the dataset
fidelities = preprocess_fidelity_df(pd.read_csv("fidelity.csv"))
baselines = preprocess_baselines_df(pd.read_csv("baselines.csv"))

data = pd.concat([fidelities, baselines], ignore_index=True)

data.to_csv("temp.csv")


# Group by generation_strategy and dataset and plot
for (strategy, dataset), group in data.groupby(["generation_strategy", "dataset"]):
    plot_group(
        group, dataset, strategy, title=f"reports/plot/figs/{dataset}_{strategy}.png"
    )
