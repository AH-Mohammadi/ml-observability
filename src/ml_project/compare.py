"""Model comparison and final test evaluation.

Run with:
    python -m ml_project.compare

Trains each model type in MODEL_BUILDERS on the training set, compares
them on the VALIDATION set only, selects the best by ROC-AUC (a threshold-
independent ranking metric, sensible here since churn is imbalanced and we
haven't committed to an operating threshold yet), and — only after that
selection is made — evaluates the single selected pipeline on the held-out
TEST set exactly once.

This ordering is deliberate: comparing all models on the test set and
picking the best there would leak information from the test set into model
selection, making the final reported number optimistic. See README
"ML pipeline" section for the training/validation/test discipline this
project follows.
"""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from ml_project.data import TARGET_COLUMN, load_data
from ml_project.evaluate import evaluate
from ml_project.split import split_data
from ml_project.train import MODEL_BUILDERS, EXPERIMENT_NAME, build_pipeline
from ml_project.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml_project.tracking import configure_default_tracking

SELECTION_METRIC = "roc_auc"


def compare_models(data_path: Path | str | None = None) -> pd.DataFrame:
    """Train and evaluate every model type on the validation set.

    Returns a DataFrame with one row per model type, sorted best-first by
    SELECTION_METRIC. Does not touch the test set.
    """
    df = load_data(data_path)
    splits = split_data(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train, y_train = splits.train[feature_cols], splits.train[TARGET_COLUMN]
    X_val, y_val = splits.val[feature_cols], splits.val[TARGET_COLUMN]

    rows = []
    fitted_pipelines = {}

    for model_type in MODEL_BUILDERS:
        pipeline = build_pipeline(model_type)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_val, y_val)
        rows.append({"model_type": model_type, **metrics})
        fitted_pipelines[model_type] = pipeline

    results = pd.DataFrame(rows).sort_values(SELECTION_METRIC, ascending=False).reset_index(drop=True)
    return results, fitted_pipelines, splits, feature_cols


def run_comparison(data_path: Path | str | None = None) -> dict:
    """Compare models on validation, select the best, evaluate it once on test."""
    results, fitted_pipelines, splits, feature_cols = compare_models(data_path)

    print("Validation comparison (sorted by {}):".format(SELECTION_METRIC))
    print(results.to_string(index=False))

    best_model_type = results.iloc[0]["model_type"]
    best_pipeline = fitted_pipelines[best_model_type]

    X_test, y_test = splits.test[feature_cols], splits.test[TARGET_COLUMN]
    test_metrics = evaluate(best_pipeline, X_test, y_test)

    configure_default_tracking()
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.set_tag("stage", "final_test_evaluation")
        mlflow.log_param("selected_model_type", best_model_type)
        mlflow.log_param("selection_metric", SELECTION_METRIC)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

    output = {
        "validation_comparison": results.to_dict(orient="records"),
        "selected_model_type": best_model_type,
        "selection_metric": SELECTION_METRIC,
        "final_test_metrics": test_metrics,
    }
    print("\nSelected model (by validation {}): {}".format(SELECTION_METRIC, best_model_type))
    print("Final test evaluation (run once, on held-out test set):")
    print(json.dumps(test_metrics, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compare models on validation, then evaluate the best once on test."
    )
    parser.add_argument("--data-path", default=None, help="Path to the raw CSV.")
    args = parser.parse_args()
    run_comparison(data_path=args.data_path)


if __name__ == "__main__":
    main()
