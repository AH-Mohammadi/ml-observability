import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_project.data import TARGET_COLUMN
from ml_project.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor
from ml_project.simulate_drift import apply_drift_shift, run_incident_simulation


@pytest.fixture
def synthetic_csv_and_model(tmp_path):
    """A synthetic dataset + a fitted model saved to disk, large enough for
    a meaningful split and drift comparison."""
    rng = np.random.default_rng(20)
    n = 600
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.uniform(20, 120, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.3, 0.35, 0.35])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
        p=[0.2, 0.2, 0.3, 0.3],
    )
    churn_prob = np.clip(0.5 - tenure / 150 + (contract == "Month-to-month") * 0.25, 0.05, 0.9)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "Contract": contract,
            "PaymentMethod": payment,
            TARGET_COLUMN: np.where(churn == 1, "Yes", "No"),
        }
    )
    csv_path = tmp_path / "synthetic_telco.csv"
    df.to_csv(csv_path, index=False)

    pipeline = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET_COLUMN])
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    return csv_path, model_path


def test_apply_drift_shift_moves_contract_toward_month_to_month():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "Contract": ["One year"] * 100,
            "PaymentMethod": ["Mailed check"] * 100,
            "MonthlyCharges": [50.0] * 100,
            "tenure": [30] * 100,
        }
    )
    shifted = apply_drift_shift(df, rng)
    mtm_rate = (shifted["Contract"] == "Month-to-month").mean()
    assert mtm_rate > 0.5  # shifted distribution has p=0.75 for month-to-month


def test_apply_drift_shift_increases_monthly_charges():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "Contract": ["One year"] * 50,
            "PaymentMethod": ["Mailed check"] * 50,
            "MonthlyCharges": [50.0] * 50,
            "tenure": [30] * 50,
        }
    )
    shifted = apply_drift_shift(df, rng)
    assert (shifted["MonthlyCharges"] > df["MonthlyCharges"]).all()


def test_incident_simulation_detects_drift(synthetic_csv_and_model):
    csv_path, model_path = synthetic_csv_and_model
    report = run_incident_simulation(data_path=csv_path, model_path=model_path)

    assert report["incident_detected"] is True
    assert "Contract" in report["drifted_features"]
    assert "PaymentMethod" in report["drifted_features"]


def test_incident_simulation_normal_batch_has_no_drift(synthetic_csv_and_model):
    csv_path, model_path = synthetic_csv_and_model
    report = run_incident_simulation(data_path=csv_path, model_path=model_path)

    normal_drifted = [r.feature for r in report["normal_batch_results"] if r.drifted]
    assert normal_drifted == []


def test_incident_simulation_reproducible_with_same_seed(synthetic_csv_and_model):
    csv_path, model_path = synthetic_csv_and_model
    report1 = run_incident_simulation(data_path=csv_path, model_path=model_path, seed=42)
    report2 = run_incident_simulation(data_path=csv_path, model_path=model_path, seed=42)

    stats1 = [r.statistic for r in report1["shifted_batch_results"]]
    stats2 = [r.statistic for r in report2["shifted_batch_results"]]
    assert stats1 == stats2


def test_missing_model_raises_clear_error(synthetic_csv_and_model, tmp_path):
    csv_path, _ = synthetic_csv_and_model
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        run_incident_simulation(data_path=csv_path, model_path=tmp_path / "nope.joblib")
