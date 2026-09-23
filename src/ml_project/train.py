"""CLI training entrypoint.

Run with:
    python -m ml_project.train

Loads data, splits it, trains a logistic regression baseline in a
Pipeline (preprocessing + model), evaluates on the validation set, and
saves the fitted pipeline to disk.
"""

import argparse
import json
from pathlib import Path

import joblib
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


def build_pipeline() -> Pipeline:
    """Build the full preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
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


def run_training(data_path: Path | str | None = None, model_path: Path | str | None = None) -> dict:
    """Run the full training flow. Returns the validation metrics dict."""
    df = load_data(data_path)
    splits = split_data(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = splits.train[feature_cols], splits.train[TARGET_COLUMN]
    X_val, y_val = splits.val[feature_cols], splits.val[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_val, y_val)

    save_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)

    print(json.dumps({"validation_metrics": metrics, "model_saved_to": str(save_path)}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the churn baseline model.")
    parser.add_argument("--data-path", default=None, help="Path to the raw CSV.")
    parser.add_argument("--model-path", default=None, help="Where to save the fitted pipeline.")
    args = parser.parse_args()
    run_training(data_path=args.data_path, model_path=args.model_path)


if __name__ == "__main__":
    main()
