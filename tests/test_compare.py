import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from ml_project.compare import SELECTION_METRIC, compare_models, run_comparison
from ml_project.data import TARGET_COLUMN
from ml_project.train import MODEL_BUILDERS


@pytest.fixture
def synthetic_csv(tmp_path):
    rng = np.random.default_rng(2)
    n = 400
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.uniform(20, 120, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
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
    return csv_path


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path, monkeypatch):
    """SQLite backend for tests — the plain filesystem store (file://...)
    is in maintenance mode in recent MLflow versions and raises."""
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield


def test_compare_models_returns_one_row_per_model_type(synthetic_csv):
    results, fitted_pipelines, splits, feature_cols = compare_models(synthetic_csv)
    assert len(results) == len(MODEL_BUILDERS)
    assert set(results["model_type"]) == set(MODEL_BUILDERS.keys())


def test_comparison_sorted_best_first_by_selection_metric(synthetic_csv):
    results, *_ = compare_models(synthetic_csv)
    values = results[SELECTION_METRIC].tolist()
    assert values == sorted(values, reverse=True)


def test_run_comparison_evaluates_test_set_exactly_once(synthetic_csv, monkeypatch):
    call_count = {"n": 0}
    from ml_project import compare as compare_module

    original_evaluate = compare_module.evaluate

    def counting_evaluate(pipeline, X, y):
        result = original_evaluate(pipeline, X, y)
        # The test set is only ever length == splits.test length; validation
        # calls happen inside compare_models with the val set. We just count
        # total evaluate() calls: len(MODEL_BUILDERS) for validation + 1 for
        # the single final test evaluation.
        call_count["n"] += 1
        return result

    monkeypatch.setattr(compare_module, "evaluate", counting_evaluate)
    run_comparison(data_path=synthetic_csv)
    assert call_count["n"] == len(MODEL_BUILDERS) + 1


def test_selected_model_matches_top_of_comparison_table(synthetic_csv):
    output = run_comparison(data_path=synthetic_csv)
    assert output["selected_model_type"] == output["validation_comparison"][0]["model_type"]


def test_final_test_metrics_have_expected_keys(synthetic_csv):
    output = run_comparison(data_path=synthetic_csv)
    assert set(output["final_test_metrics"].keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
