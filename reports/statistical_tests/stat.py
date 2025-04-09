from utils.RunLogger import RunLogger
from tqdm import tqdm
from utils.utils import load_log
from reports.statistical_tests.stat_funcs import shapiro_test, t_test, wilcoxon_test
from pathlib import Path
import pandas as pd


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


baseline_item_mapping = {
    "ml-100k": {50: 358, 411: 657, 630: 785, 1305: 1387},
    "steam": {271590: 333, 35140: 2, 292140: 1223, 582160: 448},
    "ml-1m": {2858: 105, 2005: 718, 728: 248, 2738: 1761},
}


def map_generation_strategy(row):
    if row["targeted"] and row["categorized"]:
        return "targeted"
    elif row["targeted"] and not row["categorized"]:
        return "targeted_uncategorized"
    elif not row["targeted"] and not row["categorized"]:
        return "genetic"
    elif not row["targeted"] and row["categorized"]:
        return "genetic_categorized"


def map_target(row):
    dataset = row["dataset"]
    target = row["target"]
    if target == "None":
        return target
    try:
        target = int(target)
        return f"{{{baseline_item_mapping[dataset][target]}}}"
    except:
        return f"{{{baseline_categories_mapping[dataset][target]}}}"


def seed_avg(df, subset, avg_columns):
    # Identify which columns are not part of the subset or the avg_columns
    static_columns = [col for col in df.columns if col not in subset + avg_columns]
    # Group by the subset and average the specified columns
    df_avg = df.groupby(subset, as_index=False)[avg_columns].mean()
    # Take the first row in each group to preserve static info
    df_static = df.sort_values("seed").drop_duplicates(subset)[subset + static_columns]
    # Merge static columns with averaged values
    df_merged = pd.merge(df_static, df_avg, on=subset)
    return df_merged


def clean_alignment(df, ks):
    df["gen_target_y_at_1"] = df["gen_target_y_at_1"].fillna("None")
    for k in ks:
        df[f"gen_score_at_{k}"] = df[f"gen_score_at_{k}"].fillna(0.0)
    df = df.rename(columns={"gen_target_y_at_1": "target"})
    return df


def clean_baselines(df):
    df["target"] = df["target"].fillna("None")
    df["generation_strategy"] = df.apply(map_generation_strategy, axis=1)
    df["target"] = df.apply(map_target, axis=1)
    return df


def score_test(alignment_group, baseline_group, ks, logger, group_keys):
    for basename in ["random", "educated"]:
        for key, group_align in tqdm(alignment_group):
            if key not in baseline_group.groups:
                print(f"[DEBUG] Key {key} does not exist in baseline_group")
                continue

            # Take the baseline with same combination
            group_base = baseline_group.get_group(key)
            group_base = group_base[group_base["baseline"] == basename]

            for k in ks:
                align_x = set(group_align["source"].values)
                base_x = set(group_base["source"].values)
                common_xs = align_x & base_x
                values_align = []
                values_base = []
                for x in common_xs:
                    matching_align_row = group_align[group_align["source"] == x]
                    matching_base_row = group_base[group_base["source"] == x]
                    assert str(matching_align_row["source"].values[0]) == str(
                        matching_base_row["source"].values[0]
                    ), f"{str(matching_align_row["source"])} =! f{str(matching_base_row["source"])}"
                    values_align.append(
                        matching_align_row[f"gen_score_at_{k}"].values[0]
                    )
                    values_base.append(matching_base_row[f"score_at_{k}"].values[0])
                assert len(values_align) == len(
                    values_base
                ), f"{len(values_align)} != {len(values_base)} for key {key} and k {k}"

                shapiro_align = shapiro_test(values_align)
                shapiro_base = shapiro_test(values_base)
                if shapiro_align and shapiro_base:
                    significance_t = t_test(values_base, values_align)
                    wilcoxo_t, pratt_t, zsplit_t = None, None, None
                else:
                    significance_t = None
                    wilcoxo_t, pratt_t, zsplit_t = wilcoxon_test(
                        values_align, values_base
                    )
                log_entry = dict(zip(group_keys, key))
                log_entry.update(
                    {
                        "k": k,
                        "baseline": basename,
                        "t_test": significance_t,
                        "shapiro_align": shapiro_align,
                        "shapiro_base": shapiro_base,
                        "wilcoxo": wilcoxo_t,
                        "pratt_t": pratt_t,
                        "zsplit": zsplit_t,
                    }
                )
                logger.log_run(log_entry, strict=False)


def distance_test(alignment_group, baseline_group, logger, group_keys):
    for basename in ["random", "educated"]:
        for key, group_align in tqdm(alignment_group):
            if key not in baseline_group.groups:
                continue

            # Take the baseline with same combination
            group_base = baseline_group.get_group(key)
            group_base = group_base[group_base["baseline"] == basename]

            align_x = set(group_align["source"].values)
            base_x = set(group_base["source"].values)
            common_xs = align_x & base_x
            values_align = []
            values_base = []
            for x in common_xs:
                matching_align_row = group_align[group_align["source"] == x]
                matching_base_row = group_base[group_base["source"] == x]
                assert str(matching_align_row["source"].values[0]) == str(
                    matching_base_row["source"].values[0]
                ), f"{str(matching_align_row["source"])} =! f{str(matching_base_row["source"])}"
                values_align.append(matching_align_row[f"gen_cost"].values[0])
                values_base.append(matching_base_row[f"cost"].values[0])
            assert len(values_align) == len(
                values_base
            ), f"{len(values_align)} != {len(values_base)} for key {key}"

            shapiro_align = shapiro_test(values_align)
            shapiro_base = shapiro_test(values_base)
            if shapiro_align and shapiro_base:
                significance_t = t_test(values_base, values_align)
                wilcoxo_t, pratt_t, zsplit_t = None, None, None
            else:
                significance_t = None
                wilcoxo_t, pratt_t, zsplit_t = wilcoxon_test(values_align, values_base)
            log_entry = dict(zip(group_keys, key))
            log_entry.update(
                {
                    "baseline": basename,
                    "t_test": significance_t,
                    "shapiro_align": shapiro_align,
                    "shapiro_base": shapiro_base,
                    "wilcoxo": wilcoxo_t,
                    "pratt_t": pratt_t,
                    "zsplit": zsplit_t,
                }
            )
            logger.log_run(log_entry, strict=False)


def test():
    ks = [1, 5, 10, 20]
    df_alignment = load_log(str(Path("results/evaluate/clean_alignment.csv")))
    df_baselines = load_log(str(Path("results/evaluate/combined_baselines.csv")))

    df_alignment = clean_alignment(df_alignment, ks)
    df_baselines = clean_baselines(df_baselines)

    group_keys = ["dataset", "model", "generation_strategy", "target"]
    df_alignment = seed_avg(
        df_alignment,
        subset=[*group_keys, "i"],
        avg_columns=[*[f"gen_score_at_{k}" for k in ks], "gen_cost"],
    )
    alignment_group = df_alignment.groupby(group_keys)
    df_baselines = seed_avg(
        df_baselines,
        subset=[*group_keys, "i"],
        avg_columns=[*[f"score_at_{k}" for k in ks], "cost"],
    )
    baseline_group = df_baselines.groupby(group_keys)
    score_logger = RunLogger(db_path="results/evaluate/score_test.db")
    score_test(
        alignment_group=alignment_group,
        baseline_group=baseline_group,
        ks=ks,
        logger=score_logger,
        group_keys=group_keys,
    )
    distance_logger = RunLogger(db_path="results/evaluate/distance_test.db")
    distance_test(
        alignment_group=alignment_group,
        baseline_group=baseline_group,
        logger=distance_logger,
        group_keys=group_keys,
    )


if __name__ == "__main__":
    test()
