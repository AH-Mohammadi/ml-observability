"""In-memory operational metrics for the inference service.

Distinct from MLflow (which tracks training experiments) and from
structured logs (which record individual events). Metrics here answer
"how is the service doing in aggregate" — request rate, error rate,
latency — the kind of thing a dashboard would poll periodically.

Deliberately simple (a thread-safe counter, no persistence, no
percentiles) — this is the "before Prometheus" version referenced in the
project plan. Swap this out once a real metrics backend is justified.
"""

import threading


class InMemoryMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.prediction_count = 0
        self._latencies_ms: list[float] = []

    def record_request(self, success: bool, latency_ms: float) -> None:
        """Record the outcome of one request."""
        with self._lock:
            self.request_count += 1
            self._latencies_ms.append(latency_ms)
            if success:
                self.prediction_count += 1
            else:
                self.error_count += 1

    def snapshot(self) -> dict:
        """Return current counters as a plain dict, safe to serialize."""
        with self._lock:
            avg_latency = (
                round(sum(self._latencies_ms) / len(self._latencies_ms), 2)
                if self._latencies_ms
                else 0.0
            )
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "prediction_count": self.prediction_count,
                "avg_latency_ms": avg_latency,
            }

    def reset(self) -> None:
        """Reset all counters. Mainly for test isolation."""
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.prediction_count = 0
            self._latencies_ms = []


# Module-level singleton — the inference process has one metrics instance.
metrics = InMemoryMetrics()
