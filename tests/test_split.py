import numpy as np
import pandas as pd
import pytest

from ml_project.data import TARGET_COLUMN
from ml_project.split import split_data


@pytest.fixture
def synthetic_df():
    """A larger synthetic dataset so split proportions/stratification are meaningful.

    Split tests shouldn't depend on the tiny 10-row data-loading fixture —
    that fixture is for schema tests, not statistical ones.
    """
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(n)],
            "tenure": rng.integers(0, 72, size=n),
            "MonthlyCharges": rng.uniform(20, 120, size=n),
            TARGET_COLUMN: rng.choice(["Yes", "No"], size=n, p=[0.27, 0.73]),
        }
    )


def test_split_sizes_approximately_correct(synthetic_df):
    splits = split_data(synthetic_df, val_size=0.15, test_size=0.15)
    n = len(synthetic_df)
    assert abs(len(splits.train) / n - 0.70) < 0.03
    assert abs(len(splits.val) / n - 0.15) < 0.03
    assert abs(len(splits.test) / n - 0.15) < 0.03


def test_no_overlap_between_splits(synthetic_df):
    splits = split_data(synthetic_df)
    train_ids = set(splits.train["customerID"])
    val_ids = set(splits.val["customerID"])
    test_ids = set(splits.test["customerID"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_all_rows_accounted_for(synthetic_df):
    splits = split_data(synthetic_df)
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(synthetic_df)


def test_target_distribution_preserved_by_stratification(synthetic_df):
    splits = split_data(synthetic_df)
    full_rate = (synthetic_df[TARGET_COLUMN] == "Yes").mean()
    for split_df in (splits.train, splits.val, splits.test):
        split_rate = (split_df[TARGET_COLUMN] == "Yes").mean()
        assert abs(split_rate - full_rate) < 0.08


def test_reproducible_with_same_seed(synthetic_df):
    a = split_data(synthetic_df, random_state=42)
    b = split_data(synthetic_df, random_state=42)
    assert list(a.train["customerID"]) == list(b.train["customerID"])


def test_invalid_split_sizes_raise(synthetic_df):
    with pytest.raises(ValueError):
        split_data(synthetic_df, val_size=0.6, test_size=0.5)
