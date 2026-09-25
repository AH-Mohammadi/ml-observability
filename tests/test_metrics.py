from ml_project.metrics import InMemoryMetrics


def test_starts_at_zero():
    m = InMemoryMetrics()
    snap = m.snapshot()
    assert snap == {
        "request_count": 0,
        "error_count": 0,
        "prediction_count": 0,
        "avg_latency_ms": 0.0,
    }


def test_successful_request_increments_prediction_and_request_count():
    m = InMemoryMetrics()
    m.record_request(success=True, latency_ms=10.0)
    snap = m.snapshot()
    assert snap["request_count"] == 1
    assert snap["prediction_count"] == 1
    assert snap["error_count"] == 0


def test_failed_request_increments_error_and_request_count():
    m = InMemoryMetrics()
    m.record_request(success=False, latency_ms=5.0)
    snap = m.snapshot()
    assert snap["request_count"] == 1
    assert snap["error_count"] == 1
    assert snap["prediction_count"] == 0


def test_avg_latency_computed_correctly():
    m = InMemoryMetrics()
    m.record_request(success=True, latency_ms=10.0)
    m.record_request(success=True, latency_ms=20.0)
    m.record_request(success=False, latency_ms=30.0)
    snap = m.snapshot()
    assert snap["avg_latency_ms"] == 20.0


def test_reset_clears_all_counters():
    m = InMemoryMetrics()
    m.record_request(success=True, latency_ms=10.0)
    m.reset()
    assert m.snapshot() == {
        "request_count": 0,
        "error_count": 0,
        "prediction_count": 0,
        "avg_latency_ms": 0.0,
    }
