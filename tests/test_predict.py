import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_project.data import TARGET_COLUMN
from ml_project.metrics import metrics
from ml_project.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor
from ml_project.predict import PredictionError, load_model, predict

VALID_FEATURES = {
    "tenure": 12,
    "MonthlyCharges": 55.0,
    "Contract": "One year",
    "PaymentMethod": "Mailed check",
}


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def fitted_model_path(tmp_path):
    """A small, quickly-fitted pipeline saved to disk — no train.py/MLflow
    involved, since predict.py is intentionally decoupled from those."""
    rng = np.random.default_rng(5)
    n = 100
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=n),
            "MonthlyCharges": rng.uniform(20, 120, size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], size=n),
            TARGET_COLUMN: rng.choice(["Yes", "No"], size=n),
        }
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET_COLUMN])

    import joblib

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


def test_valid_prediction_returns_expected_shape(fitted_model_path):
    result = predict(VALID_FEATURES, model_path=fitted_model_path)
    assert result["prediction"] in ("Yes", "No")
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_version"] == "unknown"  # no metadata sidecar for this directly-saved fixture model


def test_missing_feature_raises_prediction_error(fitted_model_path):
    incomplete = {k: v for k, v in VALID_FEATURES.items() if k != "Contract"}
    with pytest.raises(PredictionError, match="missing required feature"):
        predict(incomplete, model_path=fitted_model_path)


def test_completely_empty_input_raises_prediction_error(fitted_model_path):
    with pytest.raises(PredictionError, match="missing required feature"):
        predict({}, model_path=fitted_model_path)


def test_unknown_categorical_value_does_not_raise(fitted_model_path):
    features = {**VALID_FEATURES, "Contract": "Some new plan type"}
    result = predict(features, model_path=fitted_model_path)
    assert result["prediction"] in ("Yes", "No")


def test_invalid_numeric_value_raises_prediction_error(fitted_model_path):
    features = {**VALID_FEATURES, "tenure": "not-a-number"}
    with pytest.raises(PredictionError, match="data quality checks"):
        predict(features, model_path=fitted_model_path)


def test_missing_model_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        predict(VALID_FEATURES, model_path=tmp_path / "does_not_exist.joblib")


def test_successful_prediction_logs_completion_event(fitted_model_path, capsys):
    predict(VALID_FEATURES, model_path=fitted_model_path)
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.strip().split("\n") if line]
    completed = [e for e in events if e.get("event") == "prediction_completed"]
    assert len(completed) == 1
    assert "request_id" in completed[0]
    assert "latency_ms" in completed[0]


def test_failed_prediction_logs_failure_event(fitted_model_path, capsys):
    with pytest.raises(PredictionError):
        predict({**VALID_FEATURES, "tenure": "bad"}, model_path=fitted_model_path)
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.strip().split("\n") if line]
    failed = [e for e in events if e.get("event") == "prediction_failed"]
    assert len(failed) == 1


def test_prediction_updates_metrics(fitted_model_path):
    predict(VALID_FEATURES, model_path=fitted_model_path)
    snap = metrics.snapshot()
    assert snap["request_count"] == 1
    assert snap["prediction_count"] == 1
    assert snap["error_count"] == 0


def test_failed_prediction_updates_error_metrics(fitted_model_path):
    with pytest.raises(PredictionError):
        predict({k: v for k, v in VALID_FEATURES.items() if k != "tenure"}, model_path=fitted_model_path)
    snap = metrics.snapshot()
    assert snap["request_count"] == 1
    assert snap["error_count"] == 1


def test_load_model_caches_across_calls(fitted_model_path):
    m1 = load_model(fitted_model_path)
    m2 = load_model(fitted_model_path)
    assert m1 is m2


def test_prediction_uses_real_model_version_when_metadata_present(fitted_model_path):
    from ml_project.reproducibility import write_model_metadata

    write_model_metadata(
        fitted_model_path,
        model_version="run-xyz789",
        model_type="logistic_regression",
        mlflow_run_id="run-xyz789",
        dataset_path="/data/telco.csv",
        n_train_rows=500,
        n_val_rows=100,
        random_seed=42,
    )
    # Bust the module-level model/metadata cache so this test's freshly
    # written sidecar is actually picked up rather than an earlier
    # test's cached (metadata-less) load of the same path.
    import ml_project.predict as predict_module

    predict_module._model_cache.clear()
    predict_module._metadata_cache.clear()

    result = predict(VALID_FEATURES, model_path=fitted_model_path)
    assert result["model_version"] == "run-xyz789"


def test_prediction_log_includes_full_traceability_chain(fitted_model_path, capsys):
    from ml_project.reproducibility import write_model_metadata
    import ml_project.predict as predict_module

    write_model_metadata(
        fitted_model_path,
        model_version="run-trace1",
        model_type="logistic_regression",
        mlflow_run_id="run-trace1",
        dataset_path="/data/telco.csv",
        n_train_rows=500,
        n_val_rows=100,
        random_seed=42,
    )
    predict_module._model_cache.clear()
    predict_module._metadata_cache.clear()

    predict(VALID_FEATURES, model_path=fitted_model_path)
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.strip().split("\n") if line]
    completed = [e for e in events if e.get("event") == "prediction_completed"][0]

    assert completed["model_version"] == "run-trace1"
    assert completed["mlflow_run_id"] == "run-trace1"
    assert completed["dataset_path"] == "/data/telco.csv"
    assert "git_commit" in completed
