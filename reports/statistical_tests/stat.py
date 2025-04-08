from utils.RunLogger import RunLogger
from tqdm import tqdm
from utils.utils import load_log
from reports.statistical_tests.stat_funcs import t_test
from pathlib import Path


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


def test(logger: RunLogger):
    df_alignment = load_log(str(Path("results/evaluate/clean_alignment.csv")))
    df_baselines = load_log(str(Path("results/evaluate/combined_baselines.csv")))

    ks = [1, 5, 10, 20]
    group_keys = ["dataset", "model", "seed", "generation_strategy", "target"]
    df_alignment["gen_target_y_at_1"] = df_alignment["gen_target_y_at_1"].fillna("None")
    alignment_group = df_alignment.rename(
        columns={"gen_target_y_at_1": "target"}
    ).groupby(group_keys)

    df_baselines = df_baselines[
        df_baselines["baseline"] == "random"
    ]  # TODO: do it also for educated

    df_baselines["target"] = df_baselines["target"].fillna("None")
    df_baselines["generation_strategy"] = df_baselines.apply(
        map_generation_strategy, axis=1
    )
    df_baselines["target"] = df_baselines.apply(map_target, axis=1)
    baseline_group = df_baselines.groupby(group_keys)

    for key, group_align in tqdm(alignment_group):
        if key not in baseline_group.groups:
            print(f"[DEBUG] Key {key} does not exist in baseline_group")
            continue
        group_base = baseline_group.get_group(key).sort_values("i")
        group_align = group_align.sort_values("i")

        for k in ks:
            align_i = set(group_align["i"].values)
            base_i = set(group_base["i"].values)
            common_i = align_i & base_i
            values_align = []
            values_base = []
            for i in common_i:
                values_align.append(
                    group_align[group_align["i"] == i][f"gen_score_at_{k}"].values[0]
                )
                values_base.append(
                    group_base[group_base["i"] == i][f"score_at_{k}"].values[0]
                )
            assert len(values_align) == len(
                values_base
            ), f"{len(values_align)} != {len(values_base)} for key {key} and k {k}"

            significance_t = t_test(values_base, values_align)
            log_entry = dict(zip(group_keys, key))
            log_entry.update(
                {
                    "seed": int(log_entry["seed"]),
                    "k": k,
                    "significance_t_test": significance_t,
                }
            )
            logger.log_run(log_entry, strict=False)


if __name__ == "__main__":
    logger = RunLogger(db_path="results/evaluate/statistical.db")
    test(logger)
