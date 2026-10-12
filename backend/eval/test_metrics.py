"""Tiny examples small enough to check by hand. Each test says how the expected answer was worked out."""

import math

import pytest

from eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

# ---------- recall@k ----------


def test_recall_finds_one_of_three():
    # Top 3 = d1, d2, d3. Relevant = d2, d4, d9. Only d2 is in the top 3 -> 1/3.
    assert recall_at_k(["d1", "d2", "d3", "d4"], {"d2": 1, "d4": 1, "d9": 1}, k=3) == pytest.approx(1 / 3)


def test_recall_grows_with_k():
    # With k=4, d4 also makes it in -> 2/3.
    assert recall_at_k(["d1", "d2", "d3", "d4"], {"d2": 1, "d4": 1, "d9": 1}, k=4) == pytest.approx(2 / 3)


def test_recall_perfect_and_zero():
    assert recall_at_k(["a", "b"], {"a": 1, "b": 1}, k=2) == 1.0
    assert recall_at_k(["x", "y"], {"a": 1}, k=2) == 0.0


# ---------- reciprocal rank (MRR) ----------


def test_reciprocal_rank_first_hit_at_rank_three():
    # d5, d8 miss; d2 at rank 3 is the first hit -> 1/3. The later hit d1 doesn't matter.
    assert reciprocal_rank(["d5", "d8", "d2", "d1"], {"d2": 1, "d1": 1}, k=10) == pytest.approx(1 / 3)


def test_reciprocal_rank_top_hit_is_one():
    assert reciprocal_rank(["a", "b", "c"], {"a": 1}, k=10) == 1.0


def test_reciprocal_rank_hit_beyond_k_counts_as_zero():
    # The only relevant doc is at rank 3, but we only look at the top 2.
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}, k=2) == 0.0


# ---------- nDCG@k ----------


def test_ndcg_worked_example_from_docstring():
    # DCG  = 0 + 1/log2(3) + 1/log2(4) = 0.631 + 0.5   = 1.131
    # IDCG = 1/log2(2) + 1/log2(3)     = 1.0   + 0.631 = 1.631
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["d1", "d2", "d3"], {"d2": 1, "d3": 1}, k=3) == pytest.approx(expected)
    assert expected == pytest.approx(0.693, abs=0.001)


def test_ndcg_perfect_ordering_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 1, "b": 1}, k=3) == pytest.approx(1.0)


def test_ndcg_rewards_putting_the_higher_grade_first():
    relevant = {"great": 2, "ok": 1}
    better = ndcg_at_k(["great", "ok"], relevant, k=2)  # perfect order -> 1.0
    worse = ndcg_at_k(["ok", "great"], relevant, k=2)
    assert better == pytest.approx(1.0)
    # worse = (1/1 + 2/log2(3)) / (2/1 + 1/log2(3)) = 2.262 / 2.631 = 0.860
    assert worse == pytest.approx(0.860, abs=0.001)


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a": 1}, k=2) == 0.0


# ---------- averaging ----------


def test_mean_of_per_query_scores():
    assert mean([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert mean([]) == 0.0
