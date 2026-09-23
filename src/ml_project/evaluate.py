"""Metric computation for churn model evaluation.

Kept separate from train.py so both the training script and the model
comparison script can compute metrics identically without importing
MLflow-specific training internals.
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate(pipeline: Pipeline, X, y) -> dict:
    """Compute standard classification metrics for a fitted pipeline.

    `pos_label="Yes"` throughout since churn is the minority/positive class
    we care about detecting.
    """
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, pos_label="Yes"), 4),
        "recall": round(recall_score(y, y_pred, pos_label="Yes"), 4),
        "f1": round(f1_score(y, y_pred, pos_label="Yes"), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
    }
