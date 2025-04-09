from pathlib import Path

from utils.utils import load_log

DEFAULT_DB_PATH = Path("results/evaluate/alignment.db")


def clean_data(db_path: Path = DEFAULT_DB_PATH):
    df = load_log(str(db_path))
    whitelist = {
        "ml-100k": [358, 657, 785, 1387],
        "steam": [333, 2, 1223, 448],
        "ml-1m": [105, 718, 248, 1761],
    }

    def is_in_whitelist(row):
        if row["generation_strategy"] != "targeted_uncategorized":
            return True
        dataset = row["dataset"]
        target = row["gen_target_y_at_1"]
        if target[0] == "{":
            target = int(target.replace("{", "").replace("}", ""))
            print(f"Target is:", target)
        else:
            target = int(target)
        return target in whitelist.get(dataset, [])

    df = df[df.apply(is_in_whitelist, axis=1)]
    return df


def main():
    clean_df = clean_data(DEFAULT_DB_PATH)
    clean_df.to_csv(str(Path("results/evaluate/clean_alignment.csv")))


if __name__ == "__main__":
    main()
