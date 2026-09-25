import numpy as np
import pandas as pd
import pytest

from ml_project.drift import (
    check_categorical_drift,
    check_numeric_drift,
    check_prediction_drift,
    population_stability_index,
    run_data_drift_report,
)


def test_psi_is_zero_for_identical_distributions():
    ref = pd.Series({"a": 0.5, "b": 0.5})
    prod = pd.Series({"a": 0.5, "b": 0.5})
    psi = population_stability_index(ref, prod)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_is_positive_and_large_for_very_different_distributions():
    ref = pd.Series({"a": 0.9, "b": 0.1})
    prod = pd.Series({"a": 0.1, "b": 0.9})
    psi = population_stability_index(ref, prod)
    assert psi > 0.25  # well above the drift threshold


@pytest.fixture
def reference_df():
    """Reference: mostly two-year/one-year contracts, mostly automatic
    payment, as in typical training data."""
    rng = np.random.default_rng(10)
    n = 500
    return pd.DataFrame(
        {
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n, p=[0.3, 0.35, 0.35]
            ),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n,
                p=[0.2, 0.2, 0.3, 0.3],
            ),
            "tenure": rng.normal(32, 15, size=n).clip(0, 72),
            "MonthlyCharges": rng.normal(64, 20, size=n).clip(20, 120),
        }
    )


@pytest.fixture
def normal_production_df():
    """A new batch drawn from the SAME distribution as reference — should
    show no drift."""
    rng = np.random.default_rng(11)
    n = 300
    return pd.DataFrame(
        {
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n, p=[0.3, 0.35, 0.35]
            ),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n,
                p=[0.2, 0.2, 0.3, 0.3],
            ),
            "tenure": rng.normal(32, 15, size=n).clip(0, 72),
            "MonthlyCharges": rng.normal(64, 20, size=n).clip(20, 120),
        }
    )


@pytest.fixture
def shifted_production_df():
    """Our churn drift story: a pricing change / competitor pushes more
    customers to month-to-month + electronic check, at higher charges."""
    rng = np.random.default_rng(12)
    n = 300
    return pd.DataFrame(
        {
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n, p=[0.75, 0.15, 0.10]
            ),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n,
                p=[0.65, 0.1, 0.15, 0.1],
            ),
            "tenure": rng.normal(32, 15, size=n).clip(0, 72),
            "MonthlyCharges": rng.normal(95, 20, size=n).clip(20, 150),
        }
    )


def test_normal_production_data_shows_no_categorical_drift(reference_df, normal_production_df):
    result = check_categorical_drift(reference_df["Contract"], normal_production_df["Contract"], "Contract")
    assert not result.drifted


def test_shifted_production_data_shows_categorical_drift(reference_df, shifted_production_df):
    result = check_categorical_drift(reference_df["Contract"], shifted_production_df["Contract"], "Contract")
    assert result.drifted
    assert result.statistic >= 0.1


def test_shifted_payment_method_also_drifts(reference_df, shifted_production_df):
    result = check_categorical_drift(
        reference_df["PaymentMethod"], shifted_production_df["PaymentMethod"], "PaymentMethod"
    )
    assert result.drifted


def test_normal_production_data_shows_no_numeric_drift(reference_df, normal_production_df):
    result = check_numeric_drift(reference_df["MonthlyCharges"], normal_production_df["MonthlyCharges"], "MonthlyCharges")
    assert not result.drifted


def test_shifted_monthly_charges_shows_numeric_drift(reference_df, shifted_production_df):
    result = check_numeric_drift(reference_df["MonthlyCharges"], shifted_production_df["MonthlyCharges"], "MonthlyCharges")
    assert result.drifted


def test_run_data_drift_report_covers_all_configured_features(reference_df, shifted_production_df):
    results = run_data_drift_report(reference_df, shifted_production_df)
    feature_names = {r.feature for r in results}
    assert feature_names == {"Contract", "PaymentMethod", "tenure", "MonthlyCharges"}


def test_prediction_drift_detects_rising_churn_rate():
    reference_predictions = ["No"] * 82 + ["Yes"] * 18  # 82/18, typical baseline
    production_predictions = ["No"] * 61 + ["Yes"] * 39  # 61/39, drifted
    result = check_prediction_drift(reference_predictions, production_predictions)
    assert result.drifted
    assert result.production_summary["Yes"] > result.reference_summary["Yes"]


def test_prediction_drift_not_flagged_for_stable_rate():
    reference_predictions = ["No"] * 82 + ["Yes"] * 18
    production_predictions = ["No"] * 80 + ["Yes"] * 20
    result = check_prediction_drift(reference_predictions, production_predictions)
    assert not result.drifted


def test_drift_result_summary_readable():
    ref = pd.Series({"a": 0.9, "b": 0.1})
    prod = pd.Series({"a": 0.1, "b": 0.9})
    result = check_categorical_drift(pd.Series(["a"] * 90 + ["b"] * 10), pd.Series(["a"] * 10 + ["b"] * 90), "test_feature")
    assert "drift detected" in result.summary()
