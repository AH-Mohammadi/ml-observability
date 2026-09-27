"""Local prediction interface.

Deliberately does not import from train.py: inference shouldn't need
MLflow installed just to serve predictions. The default model path is
duplicated here (rather than imported from train.py) for that reason.
"""

import time
import uuid
from pathlib import Path

import joblib
import pandas as pd

from ml_project.data_quality import check_request
from ml_project.logging_config import get_logger
from ml_project.metrics import metrics
from ml_project.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml_project.reproducibility import read_model_metadata

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "baseline.joblib"
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_model_cache: dict[str, object] = {}
_metadata_cache: dict[str, dict | None] = {}


class PredictionError(Exception):
    """Raised for any invalid input or failure during prediction."""


def load_model(model_path: Path | str | None = None):
    """Load (and cache) the fitted pipeline from disk."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    key = str(path)

    if key not in _model_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model found at {path}. Run `python -m ml_project.train` first."
            )
        _model_cache[key] = joblib.load(path)
        _metadata_cache[key] = read_model_metadata(path)

    return _model_cache[key]


def get_model_metadata(model_path: Path | str | None = None) -> dict | None:
    """Return the reproducibility metadata for a (cached) model, if any."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    key = str(path)
    if key not in _metadata_cache:
        load_model(model_path)  # populates the cache
    return _metadata_cache.get(key)


def predict(features: dict, model_path: Path | str | None = None) -> dict:
    """Predict churn for a single set of customer features.

    Parameters
    ----------
    features:
        Dict with keys matching REQUIRED_FEATURES (tenure, MonthlyCharges,
        Contract, PaymentMethod). Extra keys are ignored. An unseen
        categorical value (e.g. a Contract type not seen in training) is
        handled gracefully by the pipeline's OneHotEncoder(handle_unknown=
        "ignore") — it does not raise.

    Returns
    -------
    dict
        {"prediction": "Yes"|"No", "probability": float, "model_version": str}

    Raises
    ------
    PredictionError
        If a required feature is missing, or the pipeline fails on the
        given input (e.g. a non-numeric value in a numeric field).
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    quality = check_request(features)
    if quality.warnings:
        logger.info(
            "data_quality_warning",
            extra={"request_id": request_id, "warnings": quality.warnings},
        )
    if not quality.valid:
        logger.info(
            "prediction_failed",
            extra={"request_id": request_id, "error": "data_quality_check_failed", "errors": quality.errors},
        )
        metrics.record_request(success=False, latency_ms=round((time.perf_counter() - start) * 1000, 2))
        raise PredictionError(f"Request failed data quality checks: {quality.errors}")

    try:
        pipeline = load_model(model_path)
        X = pd.DataFrame([{k: features[k] for k in REQUIRED_FEATURES}])
        prediction = pipeline.predict(X)[0]
        probability = float(pipeline.predict_proba(X)[0, 1])
    except FileNotFoundError:
        # Not a prediction-input problem — a missing model is an ops issue,
        # not something the caller's input caused. Let it propagate as-is
        # rather than being relabeled as a generic PredictionError.
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "prediction_failed",
            extra={"request_id": request_id, "error": "model_not_found", "latency_ms": latency_ms},
        )
        metrics.record_request(success=False, latency_ms=latency_ms)
        raise
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "prediction_failed",
            extra={"request_id": request_id, "error": str(exc), "latency_ms": latency_ms},
        )
        metrics.record_request(success=False, latency_ms=latency_ms)
        raise PredictionError(f"Prediction failed: {exc}") from exc

    # metadata is None for a model saved before this iteration existed —
    # degrade to "unknown" rather than crashing on an older artifact.
    model_metadata = get_model_metadata(model_path)
    model_version = model_metadata["model_version"] if model_metadata else "unknown"

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    result = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "model_version": model_version,
    }

    logger.info(
        "prediction_completed",
        extra={
            "request_id": request_id,
            "model_version": model_version,
            "mlflow_run_id": model_metadata.get("mlflow_run_id") if model_metadata else None,
            "dataset_path": model_metadata.get("dataset_path") if model_metadata else None,
            "git_commit": model_metadata.get("git_commit") if model_metadata else None,
            "prediction": prediction,
            "probability": result["probability"],
            "latency_ms": latency_ms,
        },
    )
    metrics.record_request(success=True, latency_ms=latency_ms)

    return result
