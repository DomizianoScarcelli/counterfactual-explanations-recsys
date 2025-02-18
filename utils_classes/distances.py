import math
from statistics import mean
from typing import List, Set

import Levenshtein
import torch
import torch.nn.functional as F
from torch import Tensor, Value, nn

from config import ConfigParams
from constants import PADDING_CHAR
from type_hints import CategorySet


def edit_distance(t1: Tensor | list, t2: Tensor | list, normalized: bool = True):
    if normalized:
        str1, str2 = str(t1), str(t2)  # Levenshtein.ratio only works with strings
        return 1 - Levenshtein.ratio(str1, str2)
    else:
        if isinstance(t1, Tensor):
            t1 = t1.tolist()
        if isinstance(t2, Tensor):
            t2 = t2.tolist()
        if not isinstance(t1, list) or not isinstance(t2, list):
            raise ValueError(f"t1 and t2 must be tensor or lists, not {type(t1)}")
        return Levenshtein.distance(t1, t2)


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
    assert (
        PADDING_CHAR not in seq1 and PADDING_CHAR not in seq2
    ), "Sequences should not be padded when compared with self_indicator"
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


def ndcg(a: List[int], b: List[int]) -> float:
    def dcg(relevance_scores: List[float] | List[int]) -> float:
        """Calculate Discounted Cumulative Gain (DCG)."""
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    def rel(truths: List[int], preds: List[int]) -> List[float]:
        """Calculate relevance based on positional agreement."""
        rels = []
        for j, truth in enumerate(truths):
            if truth in preds:
                rank = abs(j - (preds.index(truth))) + 1
                rels.append(1 / rank)
            else:
                rels.append(0.0)
        return rels

    if len(a) != len(b):
        raise ValueError(f"Ground truth and prediction lists must have the same length: {a} and {b} with lens: {len(a)} != {len(b)}")
    
    rels = rel(a, b)
    prel = rel(a, a)

    actual_dcg = dcg(rels)
    ideal_dcg = dcg(prel)
    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    assert 0.0 <= ndcg <= 1.0, f"NDCG is not in a normalized range: {ndcg}"
    return ndcg


def intersection_weighted_ndcg(a: List[Set[int]], b: List[Set[int]]) -> float:
    """
    Calculate the NDCG for a list of ground truth sets (a)
    and predicted sets (b).
    """

    def perfect_rel(truth: set, pred: set) -> int:
        return max(len(truth), len(pred))

    def dcg(relevance_scores: List[float] | List[int]) -> float:
        """Calculate Discounted Cumulative Gain (DCG)."""
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    def rel(truth_set: Set[int], preds_set: Set[int]) -> float:
        intersection = len(truth_set & preds_set)
        # if ConfigParams.GENERATION_STRATEGY != "targeted":
        #     raise ValueError(
        #         f"intersection_weighted_ndcg must not be used in the untargeted seetting!"
        #     )
        if intersection >= 1:
            return perfect_rel(truth_set, preds_set)
        return intersection

    if len(a) != len(b):
        raise ValueError(f"Ground truth and prediction lists must have the same length: {a} and {b} with lens: {len(a)} != {len(b)}")

    # Compute relevance scores: 1 if an element of predicted_set is in truth_set, else 0
    relevance_scores = [rel(truth, pred) for truth, pred in zip(a, b)]
    # TODO: see if changing this into a relevance score of the truth with itself gives the same result.
    # This would also remove the need of the perfect_rel and just use the value 1
    ideal_relevance_score = [perfect_rel(truth, pred) for truth, pred in zip(a, b)]

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(ideal_relevance_score)
    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    assert 0.0 <= ndcg <= 1.0, f"NDCG is not in a normalized range: {ndcg}"

    return ndcg


def intersection_weighted_positional_ndcg(
    a: List[Set[int]], b: List[Set[int]]
) -> float:
    # MAJOR TODO: this has to be tested inside the genetic_categorized topk approach

    def dcg(relevance_scores: List[float] | List[int]) -> float:
        """Calculate Discounted Cumulative Gain (DCG)."""
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    def rels(truths: List[Set[int]], preds: List[Set[int]]) -> List[float]:
        rels = []
        for j, truth in enumerate(truths):
            for i, pred in enumerate(preds):
                intersection = len(truth & pred)
                if ConfigParams.GENERATION_STRATEGY == "targeted":
                    raise ValueError(
                        f"intersection_weighted_positional_ndcg must not be used in the targeted seetting!"
                    )
                    # When we are in the targeted setting, we just want the target category to be part of the output categories.
                    # E.g. if target is 10, then rel({10}, {10, 12}) should yield a perfect score since the target is in the preds set.
                    # TODO: this can be extended to be more flexible, allowing a more strict requiremenet like perfect score only if intersection is perfect.
                if intersection >= 1:
                    n_intersection = intersection / min(len(truth), len(pred))
                    rel = n_intersection * (1 / abs(j - i))
                    rels.append(rel)
            else:
                rels.append(0)
        return rels

    if len(a) != len(b):
        raise ValueError(f"Ground truth and prediction lists must have the same length: {a} and {b} with lens: {len(a)} != {len(b)}")

    # Compute relevance scores: 1 if an element of predicted_set is in truth_set, else 0
    rels = rels(a, b)
    prels = rels(a, a)

    actual_dcg = dcg(rels)
    ideal_dcg = dcg(prels)
    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    assert 0.0 <= ndcg <= 1.0
    return ndcg
