from utils import load_log
import fire
from tqdm import tqdm
import warnings

from utils_classes.distances import intersection_weighted_ndcg

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main(log_path):
    df = load_log(log_path)
    ks = [1, 5, 10, 20]
    STRATEGY = "targeted_uncategorized"

    # Ensure proper filtering
    df = df[df["generation_strategy"] == STRATEGY]

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row["gen_aligned"] is None:
            continue
        target = row["target_cat"]
        targets = {k: [{int(target)} for _ in range(k)] for k in ks}

        for k in ks:
            if k == 1:
                topk_gen_ys = [{int(row["gen_aligned_gt_at_1"])}]
                topk_ys = [{int(row["aligned_gt_at_1"])}]
            else:
                topk_gen_ys = [
                    {x} for x in map(int, row[f"gen_aligned_gt_at_{k}"].split(","))
                ]
                topk_ys = [{x} for x in map(int, row[f"aligned_gt_at_{k}"].split(","))]
            source = targets[k]
            gen_score_at_k = intersection_weighted_ndcg(source, topk_gen_ys, perfect_score=1)
            score_at_k = intersection_weighted_ndcg(source, topk_ys, perfect_score=1)
            df.loc[idx, f"gen_score_at_{k}"] = gen_score_at_k
            df.loc[idx, f"score_at_{k}"] = score_at_k

    df.to_csv(f"fixed.csv")


if __name__ == "__main__":
    fire.Fire(main)
