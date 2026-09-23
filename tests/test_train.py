import numpy as np
import pandas as pd
import pytest

from ml_project.data import TARGET_COLUMN
from ml_project.train import run_training


@pytest.fixture
def synthetic_csv(tmp_path):
    """A synthetic CSV with the full required schema, large enough for a
    meaningful stratified split (the 10-row fixture used for schema tests
    is too small for that).
    """
    rng = np.random.default_rng(1)
    n = 300
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.uniform(20, 120, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
    )
    # Make churn weakly dependent on tenure/contract so the model has
    # something real to learn, rather than pure noise.
    churn_prob = np.clip(0.5 - tenure / 150 + (contract == "Month-to-month") * 0.2, 0.05, 0.9)
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
    return csv_path


def test_run_training_returns_expected_metric_keys(synthetic_csv, tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics = run_training(data_path=synthetic_csv, model_path=model_path)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_run_training_metrics_are_valid_probabilities(synthetic_csv, tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics = run_training(data_path=synthetic_csv, model_path=model_path)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_run_training_saves_model_file(synthetic_csv, tmp_path):
    model_path = tmp_path / "model.joblib"
    run_training(data_path=synthetic_csv, model_path=model_path)
    assert model_path.exists()
