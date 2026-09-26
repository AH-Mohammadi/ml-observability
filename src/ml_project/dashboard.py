"""Centralized log search + a one-view operational dashboard.

Reads the persisted JSONL log file (see logging_config.enable_file_logging)
and the in-memory metrics snapshot, and combines them into a single view —
answering "is the service healthy right now" without grepping raw stdout.

Run with:
    python -m ml_project.dashboard
    python -m ml_project.dashboard --search event=prediction_failed
    python -m ml_project.dashboard --search request_id=abc123
"""

import json
from pathlib import Path

from ml_project.logging_config import get_log_file_path
from ml_project.metrics import metrics


def read_log_lines(log_path: Path | str | None = None) -> list[dict]:
    """Read and parse every line of the JSONL log file. Missing file ->
    empty list (nothing has been logged to disk yet), not an error."""
    path = Path(log_path) if log_path is not None else get_log_file_path()
    if not path.exists():
        return []

    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip corrupted lines rather than fail the whole read
    return lines


def search_logs(
    log_path: Path | str | None = None,
    request_id: str | None = None,
    event: str | None = None,
    model_version: str | None = None,
    error_contains: str | None = None,
) -> list[dict]:
    """Filter persisted logs by field. All provided filters are AND-ed."""
    entries = read_log_lines(log_path)

    if request_id is not None:
        entries = [e for e in entries if e.get("request_id") == request_id]
    if event is not None:
        entries = [e for e in entries if e.get("event") == event]
    if model_version is not None:
        entries = [e for e in entries if e.get("model_version") == model_version]
    if error_contains is not None:
        entries = [e for e in entries if error_contains in str(e.get("error", ""))]

    return entries


def build_dashboard_snapshot(log_path: Path | str | None = None) -> dict:
    """Combine in-memory metrics + persisted logs into one status dict."""
    entries = read_log_lines(log_path)

    completed = [e for e in entries if e.get("event") == "prediction_completed"]
    failed = [e for e in entries if e.get("event") == "prediction_failed"]
    warnings = [e for e in entries if e.get("event") == "data_quality_warning"]
    drift_events = [e for e in entries if e.get("event") == "drift_check_completed"]

    prediction_distribution: dict[str, int] = {}
    for e in completed:
        pred = str(e.get("prediction", "unknown"))
        prediction_distribution[pred] = prediction_distribution.get(pred, 0) + 1

    latest_drift_status = drift_events[-1] if drift_events else None

    return {
        "live_metrics": metrics.snapshot(),
        "logged_predictions_total": len(completed),
        "logged_failures_total": len(failed),
        "logged_data_quality_warnings_total": len(warnings),
        "prediction_distribution": prediction_distribution,
        "latest_drift_status": latest_drift_status,
    }


def print_dashboard(snapshot: dict) -> None:
    print("=== Live metrics (current process) ===")
    for k, v in snapshot["live_metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== From persisted logs (all time) ===")
    print(f"  predictions completed: {snapshot['logged_predictions_total']}")
    print(f"  predictions failed:    {snapshot['logged_failures_total']}")
    print(f"  data quality warnings: {snapshot['logged_data_quality_warnings_total']}")
    print(f"  prediction distribution: {snapshot['prediction_distribution']}")

    print("\n=== Latest drift status ===")
    drift = snapshot["latest_drift_status"]
    if drift is None:
        print("  no drift check has been logged yet — run scripts/simulate_drift.py")
    else:
        marker = "⚠️ " if drift.get("incident_detected") else "✅ "
        print(f"  {marker}incident_detected={drift.get('incident_detected')}")
        print(f"  drifted_features: {drift.get('drifted_features')}")
        print(f"  prediction_drift: {drift.get('prediction_drift')} ({drift.get('prediction_drift_severity')})")
        print(f"  as of: {drift.get('timestamp')}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Show the operational dashboard, or search logs.")
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--search",
        default=None,
        help="field=value, e.g. --search event=prediction_failed or --search request_id=abc123",
    )
    args = parser.parse_args()

    if args.search:
        field, _, value = args.search.partition("=")
        results = search_logs(log_path=args.log_file, **{field: value})
        print(f"Found {len(results)} matching log entries:")
        for entry in results:
            print(json.dumps(entry, indent=2))
        return

    snapshot = build_dashboard_snapshot(log_path=args.log_file)
    print_dashboard(snapshot)


if __name__ == "__main__":
    main()
