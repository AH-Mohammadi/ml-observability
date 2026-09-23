import pandas as pd

from ml_project.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


def _sample_df():
    return pd.DataFrame(
        {
            "tenure": [1, 34, None],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "Contract": ["Month-to-month", "One year", None],
            "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
        }
    )


def test_preprocessor_fits_and_transforms():
    df = _sample_df()
    pre = build_preprocessor()
    transformed = pre.fit_transform(df)
    assert transformed.shape[0] == len(df)


def test_preprocessor_handles_missing_values():
    df = _sample_df()
    pre = build_preprocessor()
    # Should not raise despite None in tenure and Contract.
    pre.fit_transform(df)


def test_preprocessor_handles_unseen_category_at_transform_time():
    train_df = _sample_df().dropna()
    pre = build_preprocessor()
    pre.fit(train_df)

    new_df = pd.DataFrame(
        {
            "tenure": [10],
            "MonthlyCharges": [40.0],
            "Contract": ["Two year"],  # not seen during fit
            "PaymentMethod": ["Credit card (automatic)"],  # not seen during fit
        }
    )
    # handle_unknown="ignore" means this should not raise.
    pre.transform(new_df)


def test_expected_feature_lists_are_disjoint():
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))
