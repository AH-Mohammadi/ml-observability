"""Incident simulation: reproduce a detectable drift incident locally.

Walks through the full story this project is meant to demonstrate:

    normal production data (no drift)
              |
    a pricing change / competitor event shifts customer behavior
              |
    data drift detector triggers on Contract / PaymentMethod / MonthlyCharges
              |
    prediction drift detector triggers on the churn rate
              |
    investigation summary identifies which features moved and by how much

The simulated shift is the churn drift story settled on earlier: a
pricing change or new low-cost competitor pushes more customers toward
month-to-month contracts and electronic-check payment, alongside a
monthly price increase. It's a *simulated* shift — this dataset has no
native time axis, so there's no real "before/after" to observe, only a
constructed one (see README Limitations).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_project.data import load_data
from ml_project.drift import check_prediction_drift, run_data_drift_report
from ml_project.logging_config import get_logger
from ml_project.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml_project.split import split_data

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "baseline.joblib"
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def apply_drift_shift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate the churn drift story on a copy of `df`.

    Shifts Contract toward month-to-month, PaymentMethod toward electronic
    check, and applies a ~15% + $10 price increase to MonthlyCharges —
    standing in for a pricing change or a new low-cost competitor.
    """
    shifted = df.copy()
    n = len(shifted)
    shifted["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.75, 0.15, 0.10]
    )
    shifted["PaymentMethod"] = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
        p=[0.65, 0.1, 0.15, 0.1],
    )
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 1.15 + 10
    return shifted


def run_incident_simulation(
    data_path: Path | str | None = None,
    model_path: Path | str | None = None,
    seed: int = 123,
) -> dict:
    """Run the full incident simulation. Returns a structured report dict."""
    df = load_data(data_path)
    splits = split_data(df)
    reference_df = splits.train

    rng = np.random.default_rng(seed)

    # Step 1: a normal production batch — drawn from the same distribution
    # as reference (the validation split). Should show no drift.
    normal_batch = splits.val
    normal_results = run_data_drift_report(reference_df, normal_batch)

    # Step 2: a shifted production batch — the simulated incident.
    shifted_batch = apply_drift_shift(splits.test, rng)
    shifted_results = run_data_drift_report(reference_df, shifted_batch)

    # Step 3: prediction drift, using the trained model on both batches.
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m ml_project.train` first."
        )
    pipeline = joblib.load(path)
    reference_preds = pipeline.predict(reference_df[FEATURE_COLS])
    shifted_preds = pipeline.predict(shifted_batch[FEATURE_COLS])
    prediction_drift_result = check_prediction_drift(reference_preds, shifted_preds)

    drifted_features = [r.feature for r in shifted_results if r.drifted]
    incident_detected = bool(drifted_features) or prediction_drift_result.drifted

    logger.info(
        "drift_check_completed",
        extra={
            "incident_detected": incident_detected,
            "drifted_features": drifted_features,
            "prediction_drift": prediction_drift_result.drifted,
            "prediction_drift_severity": prediction_drift_result.severity,
        },
    )

    return {
        "normal_batch_results": normal_results,
        "shifted_batch_results": shifted_results,
        "prediction_drift_result": prediction_drift_result,
        "drifted_features": drifted_features,
        "incident_detected": incident_detected,
    }


def print_report(report: dict) -> None:
    print("=== Step 1: normal production batch (should show no drift) ===")
    for r in report["normal_batch_results"]:
        print(r.summary())

    print("\n=== Step 2: shifted production batch (simulated pricing/competitor event) ===")
    for r in report["shifted_batch_results"]:
        print(r.summary())

    print("\n=== Step 3: prediction drift ===")
    print(report["prediction_drift_result"].summary())

    print("\n=== Investigation summary ===")
    if report["incident_detected"]:
        print(f"⚠️  INCIDENT: drift detected in: {report['drifted_features']}")
        print(
            "Investigate upstream: check whether Contract/PaymentMethod distributions "
            "or pricing changed recently (e.g. a pricing change or competitor promotion)."
        )
    else:
        print("✅ No incident: production data looks consistent with training data.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simulate a drift incident and report on it.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    report = run_incident_simulation(data_path=args.data_path, model_path=args.model_path, seed=args.seed)
    print_report(report)


if __name__ == "__main__":
    main()
