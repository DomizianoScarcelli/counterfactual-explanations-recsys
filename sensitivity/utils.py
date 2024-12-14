from math import log2
from typing import Counter, List, Tuple

import torch
from torch import Tensor

from config import ConfigParams
from constants import MAX_LENGTH
from generation.utils import get_category_map
from models.config_utils import generate_model, get_config
from models.utils import pad, topk

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
    """Print detailed information about the top-k predictions for a sequence.

    This function computes the top-k predictions for a given sequence using a model, maps these predictions
    to their respective categories, and displays detailed information about each category, including its
    counterfactual count and score.

    Args:
        seq (List[int]): The input sequence represented as a list of integers.
        cat_count (Counter): Counter containing category counts for the sequence.
        dscores (Counter): Counter containing score metrics for the sequence.

    Notes:
        - The sequence is padded to a predefined `MAX_LENGTH` before being passed to the model.
        - The top-k items are computed using the `topk` function, where `k` is currently set to 10.
        - Information for each top-k category includes its name, count from `cat_count`, and score from `dscores`.

    Example Output:
        ----------------------DEBUG----------------------
        Sequence: [1, 2, 3, 4]
        Cat count: Counter({'A': 0.5, 'B': 0.3})
        Scores: Counter({'score1': 1.0, 'score2': 0.8})
        Entropy: 0.6730116670092565
        Next top-10 cats are:
        0. [('A', 0.5, 1.0), ('B', 0.3, 0.8)]
        1. [('B', 0.3, 0.8), ('C', 0.1, 0.4)]
        -------------------------------------------------
    """
    padded_seq = pad(torch.tensor(seq, dtype=torch.long), MAX_LENGTH).unsqueeze(0)
    k = ConfigParams.TOPK 
    topk_items = topk(model(padded_seq), k=k, dim=1, indices=True).squeeze(0).tolist()

    cat_map = get_category_map()
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


def compute_scores(seq: Tensor) -> Tuple[Counter, Counter]:
    """Compute category counts and discounted scores for a given sequence.

    This function calculates the total count of each category in the sequence and
    assigns scores to categories based on a discount factor, which decreases for
    characters further along in the sequence.

    Args:
        seq (List[int]): The input sequence represented as a list of integers.

    Returns:
        Tuple[Counter, Counter]:
            - cat_count: A Counter with the total occurrence of each category in the sequence.
            - dscores: A Counter with the discounted scores for each category, rounded to three decimals.

    Notes:
        - The discount factor is calculated as `1 / (1 + len(seq) - idx)`, where `idx` is the position of
          the character in the sequence.
        - Discounted scores for categories are rounded to three decimal places.

    Example:
        Given `seq = [1, 2, 3]` and `cat_map = {1: ['A'], 2: ['A', 'B'], 3: ['B']}`:
        - The category count (`cat_count`) will be: Counter({'A': 2, 'B': 2})
        - The discounted scores (`dscores`) will be: Counter({'A': 1.667, 'B': 1.333})
    """

    dscores, cat_count = Counter(), Counter()  # scores * discount_factor
    cat_map = get_category_map()
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    batch_size, _ = seq.shape

    for batch in range(batch_size):
        for idx, char in enumerate(seq[batch]):
            # Add categories for the current character
            char = char.item()
            char_categories = cat_map[char]

            discount_factor = 1 / (1 + MAX_LENGTH - idx)

            for category in char_categories:
                dscores[category] += discount_factor  # type: ignore
                cat_count[category] += 1

    return cat_count, dscores


def counterfactual_scores_deltas(seq: Tensor, position: int, alphabet: Tensor):
    """Evaluate the impact of modifying a specific position in a sequence on categorical and score-based metrics.

    This function generates all possible counterfactual sequences by replacing the value at a specified position
    in the input sequence with each value from a given alphabet. It computes the category and score metrics for
    the original sequence and the modified sequences, and calculates the changes (delta) in these metrics.

    Args:
        seq (List[int]): The input sequence represented as a list of integers.
        position (int): The position in the sequence to modify. Must be a valid index within `seq`.
        alphabet (Tensor): A tensor containing the set of possible replacement values for the position.

    Returns:
        Tuple[Counter, Counter]:
            - cat_count (Counter): The normalized category counts for the counterfactual sequences.
            - dscores (Counter): The normalized score metrics for the counterfactual sequences.

    Notes:
        - The function normalizes the category and score metrics by dividing by the size of the alphabet.
        - The deltas (differences) between the original and counterfactual metrics are printed for debugging purposes.

    Raises:
        AssertionError: If the generated tensor of counterfactual sequences has an unexpected shape.
    """
    # Generate matrix of all possible changes over the position and alphabet
    og_scores = compute_scores(seq)
    x = seq.unsqueeze(0)
    x_primes = x.repeat(len(alphabet), 1)
    assert x_primes.shape == torch.Size(
        [len(alphabet), x.size(1)]
    ), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
    positions = torch.tensor([position] * len(alphabet))
    x_primes[torch.arange(len(alphabet)), positions] = alphabet

    # for cseq in x_primes:
    cat_count, dscores = compute_scores(x_primes)

    cat_count = Counter(
        {key: value / len(alphabet) for key, value in cat_count.items()}
    )  # round and normalize
    dscores = Counter(
        {key: value / len(alphabet) for key, value in dscores.items()}
    )  # round and normalize

    og_cat_count, og_dscores = og_scores

    cat_count_delta = og_cat_count - cat_count
    dscores_delta = og_dscores - dscores

    return cat_count_delta, dscores_delta
