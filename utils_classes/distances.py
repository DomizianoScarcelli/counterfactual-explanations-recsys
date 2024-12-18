from typing import Set, List
from statistics import mean
import math

import Levenshtein
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from type_hints import CategorySet


def edit_distance(t1: Tensor, t2: Tensor, normalized: bool = True):
    if normalized:
        str1, str2 = str(t1), str(t2)  # Levenshtein.ratio only works with strings
        return 1 - Levenshtein.ratio(str1, str2)
    else:
        return Levenshtein.distance(t1.tolist(), t2.tolist())


def cosine_distance(prob1: Tensor, prob2: Tensor) -> float:
    return 1 - F.cosine_similarity(prob1, prob2, dim=-1).item()


def kl_divergence(approx: Tensor, target: Tensor) -> float:
    approx = F.log_softmax(approx)
    target = F.softmax(target)
    return F.kl_div(approx.unsqueeze(0), target.unsqueeze(0), log_target=False).item()


def jensen_shannon_divergence(p_log: Tensor, q_log: Tensor) -> float:
    # source: https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/4
    kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
    p_log, q_log = p_log.view(-1, p_log.size(-1)).log_softmax(-1), q_log.view(
        -1, q_log.size(-1)
    ).log_softmax(-1)
    m = 0.5 * (p_log + q_log)
    return 0.5 * (kl(m, p_log) + kl(m, q_log)).item()


def label_indicator(prob1: Tensor, prob2: Tensor) -> float:
    label1 = prob1.argmax(-1).item()
    label2 = prob2.argmax(-1).item()
    if label1 == label2:
        return 1
    else:
        return 0


def self_indicator(seq1: Tensor, seq2: Tensor):
    assert isinstance(seq1, Tensor) and isinstance(seq2, Tensor)
    assert (
        len(seq1.shape) == 1
    ), f"Tensor should have a single dimension, shape is: {seq1.shape}"
    assert (
        len(seq2.shape) == 1
    ), f"Tensor should have a single dimension, shape is: {seq2.shape}"
    if len(seq1) != len(seq2):
        return 0
    return float("inf") if torch.all(seq1 == seq2).all() else 0


def pairwise_jaccard_sim(
    a: CategorySet | List[CategorySet], b: CategorySet | List[CategorySet]
) -> float:
    jaccards = set()
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError(
                f"set a and b must be of the same length, got: {len(a)} != {len(b)}"
            )
        for a_i, b_i in zip(a, b):
            jaccards.add(jaccard_sim(a_i, b_i))
        return mean(jaccards)
    elif isinstance(a, set) and isinstance(b, list):
        for b_i in b:
            jaccards.add(jaccard_sim(a, b_i))
        return mean(jaccards)
    elif isinstance(a, list) and isinstance(b, set):
        for a_i in a:
            jaccards.add(jaccard_sim(a_i, b))
        return mean(jaccards)
    elif isinstance(a, set) and isinstance(b, set):
        return jaccard_sim(a, b)
    else:
        raise ValueError(
            f"Parameters must be sets of ints or lists of sets of ints, not: {type(a).__name__} and {type(b).__name__}"
        )


def jaccard_sim(a: Set[int] | Tensor, b: Set[int] | Tensor) -> float:
    """
    Computes the Jaccard similarity between two sets or tensors of indices.

    Args:
        a: The first set of indices or a tensor.
        b: The second set of indices or a tensor.

    Returns:
        float: Jaccard similarity score (0.0 to 1.0).
    """
    if not isinstance(a, (Tensor, set)):
        raise TypeError(f"Expected a Tensor or Set, but got {type(a).__name__}")
    if not isinstance(b, (Tensor, set)):
        raise TypeError(f"Expected a Tensor or Set, but got {type(b).__name__}")

    b_set = set(b.tolist()) if isinstance(b, Tensor) else b
    a_set = set(a.tolist()) if isinstance(a, Tensor) else a

    intersection = len(a_set & b_set)
    union = len(a_set | b_set)

    return intersection / union if union > 0 else 0.0


def precision_at(k: int, a: Tensor, b: Tensor) -> float:
    """
    Computes the Precision@k between two ranked lists of indices.

    Args:
        k: Number of top items to consider.
        a: Tensor containing the first ranked list of indices.
        b: Tensor containing the second ranked list of indices.

    Returns:
        float: Precision@k score.
    """
    a_top_k = set(a[:k].tolist())
    b_set = set(b.tolist())
    intersection = len(a_top_k & b_set)
    return intersection / k if k > 0 else 0.0

#TODO: cambia con https://lightning.ai/docs/torchmetrics/stable/retrieval/normalized_dcg.html
def ndcg(a: List[Set[int]], b: List[Set[int]]) -> float:
    """
    Calculate the NDCG for a list of ground truth sets (a)
    and predicted sets (b).
    """

    def dcg(relevance_scores: List[float]) -> float:
        """Calculate Discounted Cumulative Gain (DCG)."""
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    def ideal_dcg(relevance_scores: List[float]) -> float:
        """Calculate Ideal DCG (IDCG)."""
        sorted_scores = sorted(relevance_scores, reverse=True)
        return dcg(sorted_scores)

    if len(a) != len(b):
        raise ValueError("Ground truth and prediction lists must have the same length.")

    total_ndcg = 0.0
    num_queries = len(a)

    for truth_set, predicted_set in zip(a, b):
        # Compute relevance scores: 1 if an element of predicted_set is in truth_set, else 0
        relevance_scores = [1 if item in truth_set else 0 for item in predicted_set]

        # DCG and IDCG
        actual_dcg = dcg(relevance_scores)
        max_dcg = ideal_dcg(relevance_scores)

        # NDCG
        ndcg = actual_dcg / max_dcg if max_dcg > 0 else 0.0
        total_ndcg += ndcg

    return total_ndcg / num_queries if num_queries > 0 else 0.0


def DEPRECATED_ndng_at(k: int, a: Tensor, b: Tensor) -> float:
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        k: Number of top items to consider.
        a: Tensor containing the ranked list of indices.
        b: Tensor containing the ground truth set of relevant indices.

    Returns:
        float: NDCG@k score.
    """
    b_set = set(b.tolist())
    gains = torch.tensor(
        [1.0 if a[i].item() in b_set else 0.0 for i in range(min(k, len(a)))]
    )
    discounts = 1 / torch.log2(torch.arange(2, k + 2).float())
    dcg = (gains[:k] * discounts).sum().item()

    ideal_gains = torch.tensor([1.0 for _ in range(min(k, len(b_set)))])
    ideal_dcg = (ideal_gains[:k] * discounts).sum().item()

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
