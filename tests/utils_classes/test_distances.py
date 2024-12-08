import torch
from torch import tensor

from utils_classes.distances import jaccard_sim, ndcg_at, precision_at

class TestJaccardSim:
    def test_JaccardSim_ReturnsCorrectValue_WhenSetsPartiallyOverlap(self):
        a = tensor([1, 2, 3])
        b = tensor([3, 4, 5])
        assert jaccard_sim(a, b) == 1 / 5, "Failed for partial overlap."

    def test_JaccardSim_ReturnsOne_WhenSetsAreIdentical(self):
        a = tensor([1, 2, 3])
        b = tensor([1, 2, 3])
        assert jaccard_sim(a, b) == 1.0, "Failed for identical sets."

    def test_JaccardSim_ReturnsZero_WhenSetsAreDisjoint(self):
        a = tensor([1, 2, 3])
        b = tensor([4, 5, 6])
        assert jaccard_sim(a, b) == 0.0, "Failed for disjoint sets."

class TestPrecisionAtK:
    def test_PrecisionAtK_ReturnsCorrectValue_WhenPartialMatchOccurs(self):
        a = tensor([1, 2, 3, 4, 5])
        b = tensor([1, 2, 6, 7, 8])
        assert precision_at(3, a, b) == 2 / 3, "Failed for partial match."

    def test_PrecisionAtK_ReturnsZero_WhenNoItemsMatch(self):
        a = tensor([1, 2, 3, 4, 5])
        b = tensor([6, 7, 8, 9, 10])
        assert precision_at(3, a, b) == 0.0, "Failed for no overlap."

    def test_PrecisionAtK_ReturnsOne_WhenAllTopKItemsMatch(self):
        a = tensor([1, 2, 3, 4, 5])
        b = tensor([1, 2, 3, 6, 7])
        assert precision_at(3, a, b) == 1.0, "Failed for perfect match."

class TestNDCGAtK:
    def test_NDCGAtK_ReturnsOne_WhenPerfectRankingIsProvided(self):
        a = tensor([1, 2, 3, 4, 5])
        b = tensor([1, 2, 3, 6, 7])
        assert torch.isclose(torch.tensor(ndcg_at(3, a, b)), tensor([1.0])), "Failed for perfect ranking."

    def test_NDCGAtK_ReturnsCorrectValue_WhenRankingIsPartial(self):
        a = tensor([1, 2, 3, 4, 5])
        b = tensor([3, 2, 1, 6, 7])
        expected_ndcg = (1 / 1 + 1 / torch.log2(tensor(3.0)) + 1 / torch.log2(tensor(4.0))) / \
                        (1 + 1 / torch.log2(tensor(3.0)) + 1 / torch.log2(tensor(4.0)))
        assert torch.isclose(torch.tensor(ndcg_at(3, a, b)), expected_ndcg), "Failed for partial ranking."

    def test_NDCGAtK_ReturnsZero_WhenNoRelevantItemsAreRanked(self):
        a = tensor([1, 2, 3, 4, 5])
        b = tensor([6, 7, 8, 9, 10])
        assert ndcg_at(3, a, b) == 0.0, "Failed for no relevant items."
