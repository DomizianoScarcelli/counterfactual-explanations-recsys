"""
Creates a csv of the type:
```
i, source, cat1, cat2, cat3, cat4, ..., catn
0, sequence0, count1, count2, count3, count4, ..., countn
1, sequence1, count1, count2, count3, count4, ..., countn
2, sequence2, count1, count2, count3, count4, ..., countn
...
```
"""

from models.utils import pad
from config import ConfigParams
from constants import MAX_LENGTH
from models.config_utils import generate_model, get_config
from typing import Counter
import fire
import torch
import pandas as pd

from genetic.utils import get_category_map

cat_map = get_category_map()
conf = get_config(dataset = ConfigParams.DATASET, model=ConfigParams.MODEL)
model = generate_model(conf)


def categorize_sequence(row):
    seq = [int(char)for char in row["source"].split(",")]
    categories = []
    for char in seq:
        categories.extend(cat_map[char])
    cat_count = Counter(categories)
    
    padded_seq = pad(torch.tensor(seq, dtype=torch.long), MAX_LENGTH).unsqueeze(0)
    next_item = model(padded_seq).argmax(-1).item()

    next_item_cat = cat_map[next_item]

    print(f"Cat count: {cat_count}")
    print(f"Next item cat is: {next_item_cat}")

def main(csv_path: str):
    df = pd.read_csv(csv_path)

    print(df)
    df.apply(categorize_sequence, axis=1)

    pass


if __name__ == "__main__":
    # python -m scripts.category_frequency_csv --csv_path="results/evaluate/alignment/different_splits_run_cats.csv"
    fire.Fire(main)
