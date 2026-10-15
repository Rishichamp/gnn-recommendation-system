"""Unit tests for the metrics, the leave-one-out split and negative sampling
(no dataset download needed)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import sample_negative, split_data  # noqa: E402
from src.evaluate import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402


def test_recall_precision():
    recs = ["A", "X", "Y", "B", "Z"]
    truth = {"A", "B", "C", "D", "E"}
    assert recall_at_k(recs, truth, 5) == pytest.approx(2 / 5)
    assert precision_at_k(recs, truth, 5) == pytest.approx(2 / 5)
    assert recall_at_k(recs, set(), 5) == 0.0


def test_ndcg_matches_docstring_example():
    recs = ["X", "A", "Y", "Z", "B"]
    ndcg = ndcg_at_k(recs, {"A", "B"}, 5)
    import math
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["A", "B"], {"A", "B"}, 2) == pytest.approx(1.0)


def test_leave_one_out_split_uses_latest_interactions():
    df = pd.DataFrame({
        "user_id":   [0, 0, 0, 0, 1, 1],
        "item_id":   [10, 11, 12, 13, 20, 21],
        "rating":    [5] * 6,
        "timestamp": [1, 2, 3, 4, 1, 2],
    })
    train, val, test = split_data(df)
    assert test[test.user_id == 0].item_id.tolist() == [13]   # latest -> test
    assert val[val.user_id == 0].item_id.tolist() == [12]     # second latest -> val
    assert sorted(train[train.user_id == 0].item_id) == [10, 11]
    # users with < 3 interactions stay entirely in train
    assert sorted(train[train.user_id == 1].item_id) == [20, 21]
    assert 1 not in set(test.user_id)


def test_negative_samples_are_unseen():
    seen = set(range(0, 95))
    negs = sample_negative(0, seen, num_items=100, n_samples=50)
    assert len(negs) == 50 and not (set(negs) & seen)
