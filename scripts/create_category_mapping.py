"""
"""

import os
import pandas as pd
import json

dataset_path = "dataset/ml-1m"
item_info_path = os.path.join(dataset_path, "ml-1m.item")

df = pd.read_csv(item_info_path, delimiter="\t")

category_map = {
    row["item_id:token"]: row["genre:token_seq"].split()
    for _, row in df.iterrows()
}


with open("category_map.json", "w") as f:
    json.dump(category_map, f, indent=2)




