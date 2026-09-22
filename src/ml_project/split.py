"""Train / validation / test splitting for the churn dataset.

Splits are stratified on the target so class balance is preserved across
all three sets, and use a fixed random seed for reproducibility.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.data import TARGET_COLUMN

RANDOM_SEED = 42


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> Splits:
    """Split a DataFrame into train/validation/test sets.

    Stratifies on `TARGET_COLUMN` so each split has a similar churn rate.
    `val_size` and `test_size` are fractions of the *original* dataset,
    not of the remainder after the first split.

    Parameters
    ----------
    df:
        The full dataset, including the target column.
    val_size:
        Fraction of the full dataset to hold out for validation.
    test_size:
        Fraction of the full dataset to hold out for testing.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    Splits
        Dataclass with `.train`, `.val`, `.test` DataFrames.
    """
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    # val_size was specified as a fraction of the *original* df, so rescale
    # it to be a fraction of the remaining train_val set.
    relative_val_size = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val[TARGET_COLUMN],
    )

    return Splits(train=train, val=val, test=test)
