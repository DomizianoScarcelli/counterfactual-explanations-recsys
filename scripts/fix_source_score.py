from generation.utils import equal_ys
import pandas as pd
import ast
from tqdm import tqdm


def fix_source_score():
    log_path = "results/evaluate/alignment/alignment_hyperopt.csv"
    df = pd.read_csv(log_path)
    original_columns = list(df.columns)

    ks = [1, 5, 10, 20]
    def fix(row):
        for k in ks:
            target_preds = row[f"gen_target_y@{k}"]
            try:
                target_preds = ast.literal_eval(target_preds)
                if isinstance(target_preds, tuple):
                    target_preds = target_preds[0]
                assert isinstance(target_preds, set), f"target_preds wrong type, {type(target_preds)}, {target_preds}"
            except ValueError:
                print(f"target preds is not valid", target_preds)
                continue
            source_preds = row[f"gen_gt@{k}"]
            try:
                source_preds = ast.literal_eval(source_preds)
                if isinstance(source_preds, tuple):
                    source_preds = source_preds[0]
                assert isinstance(source_preds, set), f"target_preds wrong type, {type(source_preds)}, {source_preds}"
            except ValueError:
                print(f"Source preds is not valid", source_preds)
                continue
            _, source_score = equal_ys(target_preds, source_preds, return_score=True)  # type: ignore
            row[f"source_score@{k}"] = str(source_score)
        return row

    tqdm.pandas() 
    df = df.progress_apply(fix, axis=1)
    df = df.astype(str)
    df.to_csv("results/evaluate/alignment/alignment_hyperopt_updated.csv", columns=original_columns)


if __name__ == "__main__":
    fix_source_score()

