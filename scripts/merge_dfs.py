import os
import re
from typing import List, Optional

import pandas as pd

from config import ConfigParams


def main(
    paths: Optional[List[str]] = None,
    dir: Optional[str] = None,
    regex: Optional[str] = None,
    primary_key: List[str] = [],
    blacklist_keys: List[str] = ["timestamp"],
    ignore_config: bool = False,
    save_path: Optional[str] = None,
):
    if not ignore_config:
        primary_key.extend(list(ConfigParams.configs_dict().keys()))
        for item in blacklist_keys:
            primary_key.remove(item)

    print(f"[DEBUG] primary key is", primary_key)
    if paths and dir:
        raise ValueError("You can only specify a list of paths or a dir, not both")
    if paths:
        dfs = [pd.read_csv(path) for path in paths]
    elif regex:
        if not dir:
            raise ValueError("If using regex, you must specify the `dir` parameter.")
        pattern = re.compile(regex)
        dfs = [
            pd.read_csv(os.path.join(dir, path))
            for path in os.listdir(dir)
            if pattern.match(path) and path.endswith(".csv")
        ]
    elif dir:
        dfs = [
            pd.read_csv(os.path.join(dir, path))
            for path in os.listdir(dir)
            if path.endswith(".csv")
        ]
    else:
        raise ValueError("At least one between paths and dir should be defined!")

    dfs = [df.astype(str) for df in dfs]
    merged_df = pd.concat(dfs, ignore_index=True).astype(str)

    assert merged_df.shape[0] == sum(df.shape[0] for df in dfs), "Sum is wrong"

    merged_df = merged_df.drop_duplicates(subset=primary_key, ignore_index=True)
    if save_path:
        merged_df.to_csv(save_path, index=False)
    else:
        print(merged_df)


# python -m cli scripts merge_dfs --primary-key="['i', 'split', 'gen_target_y@1']" --blacklist-keys="['timestamp', 'target_cat']" ---paths="['results/evaluate/alignment/alignment_hyperopt.csv', 'results/evaluate/alignment/alignment_hyperopt_untargeted.csv']" --save-path="merged_test.csv" && vd merged_test.csv
