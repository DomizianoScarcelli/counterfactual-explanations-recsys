import pandas as pd
from pandas import DataFrame
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
    df = df.rename(
        columns={"gen_target_y_at_1": "target", "avg_gen_cost_at_1": "edit_distance"}
    )
    for k in ks:
        df = df.rename(columns={f"fidelity_gen_score_at_{k}": f"fidelity@{k}"})
    df["target"] = df["target"].fillna("None")
    df = clean_data(df)
    df = df[
        [
            "generation_strategy",
            "dataset",
            "model",
            "edit_distance",
            *[f"fidelity@{k}" for k in ks],
            "target",
            "seed",
        ]
    ]
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

    df = df[
        [
            "generation_strategy",
            "dataset",
            "baseline",
            "model",
            *[f"fidelity@{k}" for k in ks],
            "target",
            "num_edits",
            "seed",
        ]
    ]
    return df


def baselines2table(df: pd.DataFrame) -> pd.DataFrame:
    subset = ["generation_strategy", "dataset", "baseline", "model", "target", "seed"]
    fidelity_cols = [f"fidelity@{k}" for k in [1, 5, 10, 20]] + ["num_edits"]
    grouped = (
        df.groupby([col for col in subset if col != "seed"])[fidelity_cols]
        .mean()
        .reset_index()
    )
    final_grouped = (
        grouped.groupby([col for col in subset if col not in ["seed", "target"]])[
            fidelity_cols
        ]
        .mean()
        .reset_index()
    )
    return final_grouped


def fidelities2table(df: pd.DataFrame) -> pd.DataFrame:
    subset = ["generation_strategy", "dataset", "model", "target", "seed"]
    fidelity_cols = [f"fidelity@{k}" for k in [1, 5, 10, 20]] + ["edit_distance"]
    grouped = (
        df.groupby([col for col in subset if col != "seed"])[fidelity_cols]
        .mean()
        .reset_index()
    )
    final_grouped = (
        grouped.groupby([col for col in subset if col not in ["seed", "target"]])[
            fidelity_cols
        ]
        .mean()
        .reset_index()
    )
    return final_grouped


def plot_group_fidelity(
    axis, df_fidelity, df_baseline, dataset, strategy, color_map, all_models
):
    k_values = [1, 5, 10, 20]
    fidelity_columns = [f"fidelity@{k}" for k in k_values]
    seed_avg_fidelity = (
        df_fidelity.groupby(["target", "model"])[fidelity_columns].mean().reset_index()
    )
    model_avg_fidelity = seed_avg_fidelity.groupby("model")[fidelity_columns].mean()

    lines = []
    # Plot models in consistent order
    for model in all_models:
        if model in model_avg_fidelity.index:
            (line,) = axis.plot(
                k_values,
                model_avg_fidelity.loc[model],
                marker="o",
                linestyle="-",
                label=model,
                color=color_map.get(model),
            )
            lines.append(line)

    df_baseline = df_baseline.sort_values("baseline", ascending=False)
    for baseline, group_baseline in df_baseline.groupby("baseline"):
        seed_avg_baseline = (
            group_baseline.groupby(["target", "model"])[fidelity_columns]
            .mean()
            .reset_index()
        )
        model_avg_baseline = seed_avg_baseline.groupby("model")[fidelity_columns].mean()

        # Plot baseline models in same order
        for model in all_models:
            if model in model_avg_baseline.index:
                baseline_label = f"({baseline}) - {model}"
                (line,) = axis.plot(
                    k_values,
                    model_avg_baseline.loc[model],
                    marker="^" if baseline == "educated" else "s",
                    linestyle="--",
                    label=baseline_label,
                    color=color_map.get(baseline_label),
                )
                lines.append(line)

    return lines


def plot_fidelities(fidelities):
    datasets = sorted(fidelities["dataset"].unique())
    strategies = sorted(fidelities["generation_strategy"].unique())

    rows = len(datasets)
    cols = len(strategies)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    # Ensure axes is always a 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, cols)
    elif cols == 1:
        axes = axes.reshape(rows, 1)

    # Create color map for consistent colors
    all_models = sorted(fidelities["model"].unique())
    color_map = {model: plt.cm.tab10(i % 10) for i, model in enumerate(all_models)}

    # Add colors for baseline versions of each model
    baseline_labels = baselines["baseline"].unique()
    for baseline in baseline_labels:
        for i, model in enumerate(all_models):
            color_map[f"({baseline}) - {model}"] = plt.cm.tab10(i % 10)

    all_lines = []
    all_labels = []

    # Plot by dataset (rows) and strategy (columns)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, strategy in enumerate(strategies):
            ax = axes[row_idx, col_idx]

            # Get data for this dataset and strategy
            dataset_strategy_data = fidelities[
                (fidelities["dataset"] == dataset)
                & (fidelities["generation_strategy"] == strategy)
            ]

            matching_baselines = baselines[
                (baselines["dataset"] == dataset)
                & (baselines["generation_strategy"] == strategy)
            ]

            if not dataset_strategy_data.empty and not matching_baselines.empty:
                lines = plot_group_fidelity(
                    ax,
                    dataset_strategy_data,
                    matching_baselines,
                    dataset,
                    strategy,
                    color_map,
                    all_models,
                )
                all_lines.extend(lines)
                all_labels.extend([line.get_label() for line in lines])

            # Add title even if there's no data
            strategy_map = {
                "genetic": "Untargeted-Uncategorized",
                "genetic_categorized": "Untargeted-Categorized",
                "targeted": "Targeted-Categorized",
                "targeted_uncategorized": "Targeted-Uncategorized",
            }
            strat = strategy_map.get(strategy, strategy)
            ax.set_title(f"{dataset} - {strat}")
            ax.set_xlabel("k")
            ax.set_ylabel("Fidelity@k")
            ax.set_xticks([1, 5, 10, 20])
            ax.grid(True)

    # Create the legend
    unique = {}
    for line, label in zip(all_lines, all_labels):
        if label not in unique:
            unique[label] = line

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig("reports/plot/figs/results.png")


baselines = preprocess_baselines_df(pd.read_csv("baselines.csv"))
fidelities = preprocess_fidelity_df(pd.read_csv("fidelity.csv"))
plot_fidelities(fidelities)

baselines = baselines2table(baselines)
baselines.to_csv(f"reports/csv/baseline_fidelities.csv")
fidelities = fidelities2table(fidelities)
fidelities.to_csv(f"reports/csv/fidelities.csv")
