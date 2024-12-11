from utils import seq_tostr
import fire
import pandas as pd
from sensitivity.utils import category_scores, print_topk_info


def scores_from_csv(csv_path: str, print_info: bool = False):
    df = pd.read_csv(csv_path)
    # just keep the first occurrence of the sequence
    df = df.drop_duplicates(subset=["i"], keep="first")

    mapping = {}
    for _, row in df.iterrows():
        header = "source"
        seq = [int(char) for char in row[header].split(",")]  # type: ignore
        cat_count, dscores = category_scores(seq)
        mapping[seq_tostr(seq)] = {"cat_count": cat_count, "dscores": dscores}

        if print_info:
            print_topk_info(seq, cat_count, dscores)

    print(f"Mapping is: {mapping}")


if __name__ == "__main__":
    # python -m scripts.csv_scores --csv_path="results/evaluate/alignment/different_splits_run_cats.csv"
    fire.Fire(scores_from_csv)
