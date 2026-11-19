import pytest
import pandas as pd
import numpy as np

from src.evaluation.walk_forward import WalkForwardSplitter


def test_walk_forward_splitter_expansion():
    # 300 mock dates
    dates = pd.date_range("2016-01-01", periods=300, freq="B").tolist()
    splitter = WalkForwardSplitter(all_dates=dates, min_train_days=150, test_days=50)

    folds = splitter.generate_folds()
    assert len(folds) >= 3

    # Fold 1
    assert folds[0]["train_count"] == 150
    assert folds[0]["test_count"] == 50
    # Strict chronological order: train_end strictly precedes test_start
    assert pd.to_datetime(folds[0]["train_end"]) < pd.to_datetime(folds[0]["test_start"])

    # Expanding window property: Fold 2 train count > Fold 1 train count
    assert folds[1]["train_count"] > folds[0]["train_count"]
    assert pd.to_datetime(folds[1]["train_end"]) < pd.to_datetime(folds[1]["test_start"])


def test_zero_lookahead_fold_isolation():
    dates = pd.date_range("2016-01-01", periods=200, freq="B").tolist()
    splitter = WalkForwardSplitter(all_dates=dates, min_train_days=100, test_days=50)
    folds = splitter.generate_folds()

    for fold in folds:
        train_set = set(fold["train_dates"])
        test_set = set(fold["test_dates"])
        # Intersection between train and test must be exactly empty
        assert len(train_set.intersection(test_set)) == 0
        # All train dates must be before any test date
        assert max(train_set) < min(test_set)
