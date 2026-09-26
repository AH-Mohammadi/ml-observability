import json

import pytest

from ml_project.dashboard import build_dashboard_snapshot, search_logs
from ml_project.metrics import metrics


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def sample_log_file(tmp_path):
    entries = [
        {"event": "prediction_completed", "request_id": "req-1", "model_version": "v1", "prediction": "Yes", "timestamp": "t1"},
        {"event": "prediction_completed", "request_id": "req-2", "model_version": "v1", "prediction": "No", "timestamp": "t2"},
        {"event": "prediction_completed", "request_id": "req-3", "model_version": "v1", "prediction": "No", "timestamp": "t3"},
        {"event": "prediction_failed", "request_id": "req-4", "error": "missing_feature: Contract", "timestamp": "t4"},
        {"event": "data_quality_warning", "request_id": "req-5", "warnings": ["unexpected category"], "timestamp": "t5"},
        {
            "event": "drift_check_completed",
            "incident_detected": True,
            "drifted_features": ["Contract", "PaymentMethod"],
            "prediction_drift": True,
            "prediction_drift_severity": "major",
            "timestamp": "t6",
        },
    ]
    log_path = tmp_path / "app.jsonl"
    with open(log_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return log_path


def test_search_by_request_id(sample_log_file):
    results = search_logs(log_path=sample_log_file, request_id="req-2")
    assert len(results) == 1
    assert results[0]["prediction"] == "No"


def test_search_by_event(sample_log_file):
    results = search_logs(log_path=sample_log_file, event="prediction_completed")
    assert len(results) == 3


def test_search_by_error_substring(sample_log_file):
    results = search_logs(log_path=sample_log_file, error_contains="missing_feature")
    assert len(results) == 1
    assert results[0]["request_id"] == "req-4"


def test_search_combined_filters(sample_log_file):
    results = search_logs(log_path=sample_log_file, event="prediction_completed", model_version="v1")
    assert len(results) == 3


def test_search_no_matches_returns_empty_list(sample_log_file):
    results = search_logs(log_path=sample_log_file, request_id="does-not-exist")
    assert results == []


def test_search_missing_log_file_returns_empty_list(tmp_path):
    results = search_logs(log_path=tmp_path / "nonexistent.jsonl")
    assert results == []


def test_dashboard_snapshot_aggregates_logs(sample_log_file):
    snapshot = build_dashboard_snapshot(log_path=sample_log_file)
    assert snapshot["logged_predictions_total"] == 3
    assert snapshot["logged_failures_total"] == 1
    assert snapshot["logged_data_quality_warnings_total"] == 1
    assert snapshot["prediction_distribution"] == {"Yes": 1, "No": 2}


def test_dashboard_snapshot_includes_latest_drift_status(sample_log_file):
    snapshot = build_dashboard_snapshot(log_path=sample_log_file)
    assert snapshot["latest_drift_status"]["incident_detected"] is True
    assert snapshot["latest_drift_status"]["drifted_features"] == ["Contract", "PaymentMethod"]


def test_dashboard_snapshot_no_drift_status_when_none_logged(tmp_path):
    empty_log = tmp_path / "empty.jsonl"
    empty_log.write_text("")
    snapshot = build_dashboard_snapshot(log_path=empty_log)
    assert snapshot["latest_drift_status"] is None


def test_dashboard_snapshot_includes_live_metrics(sample_log_file):
    metrics.record_request(success=True, latency_ms=15.0)
    snapshot = build_dashboard_snapshot(log_path=sample_log_file)
    assert snapshot["live_metrics"]["request_count"] == 1
