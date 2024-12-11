from math import log2
from typing import Counter, List

import torch

from config import ConfigParams
from constants import MAX_LENGTH
from generation.utils import get_category_map
from models.config_utils import generate_model, get_config
from models.utils import pad, topk

cat_map = get_category_map()
conf = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
model = generate_model(conf)


def entropy(counter: Counter) -> float:
    # High entropy: The sequence is more uniform and sensitive to changes in the last items.
    # Low entropy: The sequence is concentrated on specific categories and less sensitive.
    total = sum(counter.values())
    probabilities = [count / total for count in counter.values()]
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy


def print_topk_info(seq: List[int], cat_count: Counter, dscores: Counter):
    padded_seq = pad(torch.tensor(seq, dtype=torch.long), MAX_LENGTH).unsqueeze(0)
    k = 10  # TODO: make k a parameter
    topk_items = topk(model(padded_seq), k=k, dim=1, indices=True).squeeze(0).tolist()

    topk_items_cat = [cat_map[item] for item in topk_items]
    topk_items_cat_w_info = [
        [(inner, cat_count[inner], dscores[inner]) for inner in item]
        for item in topk_items_cat
    ]

    print("-" * 22 + "DEBUG" + "-" * 22)
    print(f"Sequence: {seq}")
    print(f"Cat count: {cat_count}")
    print(f"Scores: {dscores}")
    print(f"Entropy: {entropy(cat_count)}")
    print(f"Next top-{k} cats are")
    for i, tup in enumerate(topk_items_cat_w_info):
        print(f"{i}. {tup}")
    print("-" * 49)


def category_scores(seq: List[int]):
    dscores = Counter()  # scores * discount_factor
    categories = []
    for idx, char in enumerate(seq):
        # Add categories for the current character
        char_categories = cat_map[char]
        categories.extend(char_categories)

        discount_factor = 1 / (1 + len(seq) - idx)

        for category in char_categories:
            dscores[category] += discount_factor  # type: ignore

    cat_count = Counter(categories)
    dscores = Counter({key: round(value, 3) for key, value in dscores.items()})

    return cat_count, dscores
