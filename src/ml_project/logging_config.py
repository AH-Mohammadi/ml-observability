"""Structured JSON logging.

Every log line is a single JSON object, so logs can be searched/filtered
by field (event, run_id, model_type, ...) rather than by substring match
on free-text messages. This is what later gets fed into a log aggregator
(Loki) and what makes an incident debuggable from logs alone.

Usage:
    logger = get_logger(__name__)
    logger.info("training_started", extra={"run_id": run_id, "model_type": model_type})

The `event` field of each JSON line is the log message itself — callers
should pass a short, consistent event name as the message (e.g.
"training_started", "prediction_completed"), and put everything else in
`extra`.
"""

import json
import logging
import sys
from datetime import datetime, timezone

_CONFIGURED_LOGGERS: set[str] = set()

# Attributes every logging.LogRecord has by default — used to detect which
# attributes on a record were added via `extra={...}` by the caller.
_STANDARD_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Include any caller-supplied extra fields (run_id, model_type, etc.)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key != "message":
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class _CurrentStdoutHandler(logging.StreamHandler):
    """A StreamHandler that always writes to the *current* sys.stdout.

    Plain `logging.StreamHandler(sys.stdout)` binds to whatever sys.stdout
    object exists at handler-creation time (typically module import time).
    If something later swaps sys.stdout — pytest's capsys fixture, for
    instance — this handler would keep writing to the old, no-longer-
    captured stream, and tests asserting on captured log output would see
    nothing even though the code is working correctly.
    """

    def __init__(self):
        super().__init__(stream=sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass  # ignore: base class __init__ assigns self.stream once; we always resolve dynamically


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured to emit single-line JSON to stdout.

    Safe to call repeatedly with the same name (e.g. across test runs)
    without accumulating duplicate handlers.
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED_LOGGERS:
        handler = _CurrentStdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
