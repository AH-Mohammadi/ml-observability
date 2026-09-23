"""CLI training entrypoint with MLflow experiment tracking.

Run with:
    python -m ml_project.train
    python -m ml_project.train --model-type random_forest

Loads data, splits it, trains a model in a Pipeline (preprocessing +
model), evaluates on the validation set, and logs the run — params,
metrics, duration, dataset info, and the model artifact — to MLflow.
Each invocation creates a new, separately-comparable run.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ml_project.data import TARGET_COLUMN, load_data
from ml_project.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)
from ml_project.split import RANDOM_SEED, split_data

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "baseline.joblib"
EXPERIMENT_NAME = "telco-churn"

ModelType = Literal["logistic_regression", "random_forest"]

MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_SEED
    ),
}


def build_pipeline(model_type: ModelType = "logistic_regression") -> Pipeline:
    """Build the full preprocessing + model pipeline for the given model type."""
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(MODEL_BUILDERS)}.")
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", MODEL_BUILDERS[model_type]()),
        ]
    )


def evaluate(pipeline: Pipeline, X_val, y_val) -> dict:
    """Compute validation metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred, pos_label="Yes"), 4),
        "recall": round(recall_score(y_val, y_pred, pos_label="Yes"), 4),
        "f1": round(f1_score(y_val, y_pred, pos_label="Yes"), 4),
        "roc_auc": round(roc_auc_score(y_val, y_proba), 4),
    }


def run_training(
    data_path: Path | str | None = None,
    model_path: Path | str | None = None,
    model_type: ModelType = "logistic_regression",
) -> dict:
    """Run the full training flow inside an MLflow run. Returns validation metrics."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        start = time.perf_counter()

        df = load_data(data_path)
        splits = split_data(df)

        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X_train, y_train = splits.train[feature_cols], splits.train[TARGET_COLUMN]
        X_val, y_val = splits.val[feature_cols], splits.val[TARGET_COLUMN]

        pipeline = build_pipeline(model_type)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_val, y_val)
        duration_seconds = round(time.perf_counter() - start, 3)

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_train_rows", len(splits.train))
        mlflow.log_param("n_val_rows", len(splits.val))
        mlflow.log_param("n_test_rows", len(splits.test))
        mlflow.log_param("features", feature_cols)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_metric("training_duration_seconds", duration_seconds)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

        save_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)

        result = {
            "model_type": model_type,
            "validation_metrics": metrics,
            "training_duration_seconds": duration_seconds,
            "model_saved_to": str(save_path),
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        print(json.dumps(result, indent=2))
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the churn model with MLflow tracking.")
    parser.add_argument("--data-path", default=None, help="Path to the raw CSV.")
    parser.add_argument("--model-path", default=None, help="Where to save the fitted pipeline.")
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=list(MODEL_BUILDERS),
        help="Which model to train.",
    )
    args = parser.parse_args()
    run_training(data_path=args.data_path, model_path=args.model_path, model_type=args.model_type)


if __name__ == "__main__":
    main()
