import pytest
from torch import tensor

from config import ConfigParams
from utils_classes.distances import (intersection_weighted_ndcg, jaccard_sim,
                                     precision_at)


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


class TestNDCG:
    def test_NDCG_ReturnsOne_WhenListsAreIdentical(self):
        a = [{1, 2, 3}, {4, 5}]
        b = [{1, 2, 3}, {4, 5}]
        result = intersection_weighted_ndcg(a, b)
        assert pytest.approx(result, 0.00001) == 1.0, result

    def test_NDCG_ReturnsZero_WhenListsAreDisjoint(self):
        a = [{1, 2, 3}, {4, 5}]
        b = [{6, 7}, {8, 9}]
        result = intersection_weighted_ndcg(a, b)
        assert pytest.approx(result, 0.00001) == 0.0, result

    def test_NDCG_ReturnsCorrectValue_WhenPartialOverlap(self):
        a = [{1, 2}, {3, 4}]
        b = [{1, 3}, {2, 4}]
        result = intersection_weighted_ndcg(a, b)

        if ConfigParams.GENERATION_STRATEGY == "targeted":
            assert pytest.approx(result, 0.00001) == 1.0, result
        else:
            assert 0.0 < result < 1.0, result

    def test_NDCG_ReturnsOne_WhenRankingIsPerfect(self):
        a = [{1, 2}, {3}]
        b = [{1, 2}, {3}]
        result = intersection_weighted_ndcg(a, b)
        assert pytest.approx(result, 0.00001) == 1.0, result

    def test_NDCG_ReturnsLowerValue_WhenRankingIsSuboptimal(self):
        a = [{1, 2}, {3}]
        b = [{1, 3}, {3}]
        result_perfect = intersection_weighted_ndcg(a, [{1, 2}, {3}])
        result_suboptimal = intersection_weighted_ndcg(a, b)

        if ConfigParams.GENERATION_STRATEGY == "targeted":
            assert result_perfect == result_suboptimal
        else:
            assert result_perfect > result_suboptimal, result_suboptimal 

    def test_NDCG_ReturnsZero_WhenNoItemsInB(self):
        a = [{1, 2, 3}, {4, 5}]
        b = [set(), set()]
        result = intersection_weighted_ndcg(a, b)
        assert pytest.approx(result, 0.00001) == 0.0, result

    def test_NDCG_HandlesEmptyInputLists(self):
        a = []
        b = []
        result = intersection_weighted_ndcg(a, b)
        assert pytest.approx(result, 0.00001) == 0.0, result

    def test_NDCG_HandlesSingleSetInputs(self):
        a = [{1, 2}]
        b = [{3, 1}]
        result = intersection_weighted_ndcg(a, b)
        assert 0.0 < result <= 1.0, result

    def test_NDCG_HandlesVaryingLengthsOfAAndB(self):
        a = [{1, 2, 3}]
        b = [{1, 2}]
        result = intersection_weighted_ndcg(a, b)
        if ConfigParams.GENERATION_STRATEGY == "targeted":
            assert pytest.approx(result, 0.00001) == 1.0, result
        else:
            assert 0.0 < result < 1.0, result

    def test_NDCG_ReturnsOne_WhenAllItemsMatchWithPerfectRanking(self):
        a = [{1}, {2}, {3}]
        b = [{1}, {2}, {3}]
        result = intersection_weighted_ndcg(a, b)
        assert pytest.approx(result, 0.00001) == 1.0, result
