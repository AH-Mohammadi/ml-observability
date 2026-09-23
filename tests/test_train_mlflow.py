import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from ml_project.data import TARGET_COLUMN
from ml_project.train import run_training


@pytest.fixture
def synthetic_csv(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.uniform(20, 120, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
    )
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


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path, monkeypatch):
    """Point MLflow at a temp local directory so tests don't pollute the
    real `mlruns/` folder or depend on a running tracking server."""
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    yield


def test_training_creates_a_logged_run(synthetic_csv, tmp_path):
    model_path = tmp_path / "model.joblib"
    run_training(data_path=synthetic_csv, model_path=model_path, model_type="logistic_regression")

    runs = mlflow.search_runs(experiment_names=["telco-churn"])
    assert len(runs) == 1


def test_run_logs_expected_params_and_metrics(synthetic_csv, tmp_path):
    model_path = tmp_path / "model.joblib"
    run_training(data_path=synthetic_csv, model_path=model_path, model_type="logistic_regression")

    runs = mlflow.search_runs(experiment_names=["telco-churn"])
    row = runs.iloc[0]
    assert row["params.model_type"] == "logistic_regression"
    assert "metrics.roc_auc" in row
    assert "metrics.training_duration_seconds" in row


def test_two_model_types_produce_two_comparable_runs(synthetic_csv, tmp_path):
    run_training(
        data_path=synthetic_csv,
        model_path=tmp_path / "lr.joblib",
        model_type="logistic_regression",
    )
    run_training(
        data_path=synthetic_csv,
        model_path=tmp_path / "rf.joblib",
        model_type="random_forest",
    )

    runs = mlflow.search_runs(experiment_names=["telco-churn"])
    assert len(runs) == 2
    assert set(runs["params.model_type"]) == {"logistic_regression", "random_forest"}


def test_invalid_model_type_raises(synthetic_csv, tmp_path):
    with pytest.raises(ValueError, match="Unknown model_type"):
        run_training(data_path=synthetic_csv, model_path=tmp_path / "m.joblib", model_type="not_a_model")
