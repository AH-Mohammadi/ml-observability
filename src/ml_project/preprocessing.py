"""Preprocessing pipeline for the churn dataset.

Defines which columns are treated as numeric vs. categorical, and builds
a scikit-learn ColumnTransformer that can be composed with a model into a
single Pipeline. Keeping this in one place means preprocessing is always
fit on train data only, never on validation or test data (the Pipeline
enforces that: `.fit()` is called once, on train; val/test only ever see
`.transform()` via `.predict()`).
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]

CATEGORICAL_FEATURES = ["Contract", "PaymentMethod"]

# Columns intentionally excluded from the baseline: customerID (identifier,
# not a feature), TotalCharges (has a handful of blank strings for
# zero-tenure customers in the raw data — needs cleaning, deferred), and
# the remaining categorical service-add-on columns (Iteration 6+ concern
# once we're comparing models, not needed to prove the pipeline works).


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer for numeric + categorical churn features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
