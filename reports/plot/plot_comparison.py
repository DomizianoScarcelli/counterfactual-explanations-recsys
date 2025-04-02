import pandas as pd
import matplotlib.pyplot as plt

def clean_data(df):
    whitelist = {
        "ml-100k": [358, 657, 785, 1387],
        "steam": [333, 2, 1223, 448],
        "ml-1m": [105, 718, 248, 1761],
    }

    def is_in_whitelist(row):
        if row["generation_strategy"] != "targeted_uncategorized":
            return True
        dataset = row["dataset"]
        target = int(row["target"])
        return target in whitelist.get(dataset, [])

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
    df = df[["generation_strategy", "dataset", "model", *[f"fidelity@{k}" for k in ks], "target", "seed"]]
    return df

def preprocess_baselines_df(df):
    baseline_item_mapping = {
        "ml-100k": {50: 358, 411: 657, 630: 785, 1305: 1387},
        "steam": {271590: 333, 35140: 2, 292140: 1223, 582160: 448},
        "ml-1m": {2858: 105, 2005: 718, 728: 248, 2738: 1761},
    }
    baseline_categories_mapping = {
        "steam": {
            "Indie": 2,
            "Photo Editing": 6,
            "Action": 10,
            "Sports": 18,
            "Free to Play": 20,
        },
        "ml-100k": {
            "Action": 0,
            "Adventure": 1,
            "Animation": 2,
            "Drama": 7,
            "Fantasy": 8,
            "Horror": 10,
        },
        "ml-1m": {
            "Action": 0,
            "Adventure": 1,
            "Animation": 2,
            "Drama": 7,
            "Fantasy": 8,
            "Horror": 10,
        },
    }

    ks = [1, 5, 10, 20]

    for k in ks:
        df = df.rename(columns={f"fidelity_score_at_{k}": f"fidelity@{k}"})
    df["target"] = df["target"].fillna("None")

    def map_target(row):
        dataset = row["dataset"]
        target = row["target"]
        if target == "None":
            return target
        try:
            target = int(target)
            return baseline_item_mapping[dataset][target]
        except:
            return baseline_categories_mapping[dataset][target]

    df["target"] = df.apply(map_target, axis=1)

    def map_generation_strategy(row):
        if row["targeted"] and row["categorized"]:
            return "targeted"
        elif row["targeted"] and not row["categorized"]:
            return "targeted_uncategorized"
        elif not row["targeted"] and not row["categorized"]:
            return "genetic"
        elif not row["targeted"] and row["categorized"]:
            return "genetic_categorized"

    df["generation_strategy"] = df.apply(map_generation_strategy, axis=1)

    df = df[["generation_strategy", "dataset", "baseline", "model", *[f"fidelity@{k}" for k in ks], "target", "seed"]]
    return df

def plot_group(df_fidelity, df_baseline, dataset, strategy, title: str):
    plt.figure(figsize=(8, 5))
    k_values = [1, 5, 10, 20]
    fidelity_columns = [f"fidelity@{k}" for k in k_values]

    seed_avg_fidelity = df_fidelity.groupby(["target", "model"])[fidelity_columns].mean().reset_index()
    model_avg_fidelity = seed_avg_fidelity.groupby("model")[fidelity_columns].mean()

    for model in model_avg_fidelity.index:
        plt.plot(k_values, model_avg_fidelity.loc[model], marker="o", linestyle="-", label=f"Fidelity - {model}")

    for baseline, group_baseline in df_baseline.groupby("baseline"):
        seed_avg_baseline = group_baseline.groupby(["target", "model"])[fidelity_columns].mean().reset_index()
        model_avg_baseline = seed_avg_baseline.groupby("model")[fidelity_columns].mean()

        for model in model_avg_baseline.index:
            plt.plot(
                k_values,
                model_avg_baseline.loc[model],
                marker="s",
                linestyle="--",
                label=f"Baseline ({baseline}) - {model}",
            )

    plt.xlabel("k (Fidelity@k)")
    plt.ylabel("Fidelity Score")
    plt.title(f"Fidelity Scores for {dataset} - {strategy}")
    plt.legend(title="Model")
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(title, bbox_inches="tight")
    plt.close()

fidelities = preprocess_fidelity_df(pd.read_csv("fidelity.csv"))
baselines = preprocess_baselines_df(pd.read_csv("baselines.csv"))

for (strategy, dataset), group_fidelity in fidelities.groupby(["generation_strategy", "dataset"]):
    matching_baselines = baselines[(baselines["generation_strategy"] == strategy) & (baselines["dataset"] == dataset)]
    if not matching_baselines.empty:
        plot_group(
            group_fidelity,
            matching_baselines,
            dataset,
            strategy,
            title=f"reports/plot/figs/{dataset}_{strategy}_combined.png",
        )
