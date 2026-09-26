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
import threading
from datetime import datetime, timezone
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()

DEFAULT_LOG_FILE_PATH = Path(__file__).resolve().parents[2] / "logs" / "app.jsonl"

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


_file_logging_state = {"enabled": False, "path": DEFAULT_LOG_FILE_PATH}
_file_write_lock = threading.Lock()


def enable_file_logging(path: Path | str | None = None) -> None:
    """Enable persisting structured logs to a JSONL file, in addition to
    stdout. Call this once from a CLI entrypoint's main() — not from
    library code — so importing/testing a module doesn't incur file I/O.

    A global switch (rather than per-logger config) because it needs to
    take effect for every module's logger, including ones already created
    via `logger = get_logger(__name__)` at import time — see
    _CurrentStdoutHandler's docstring for the same import-order issue.
    """
    if path is not None:
        _file_logging_state["path"] = Path(path)
    _file_logging_state["path"].parent.mkdir(parents=True, exist_ok=True)
    _file_logging_state["enabled"] = True


def disable_file_logging() -> None:
    """Disable file logging. Mainly for test teardown, since the enabled
    flag is process-global."""
    _file_logging_state["enabled"] = False


def get_log_file_path() -> Path:
    return _file_logging_state["path"]


class _OptionalFileHandler(logging.Handler):
    """No-ops unless enable_file_logging() has been called.

    Every logger carries this handler; it costs nothing when file logging
    is disabled (the default, including during tests), and starts writing
    JSONL lines the moment a CLI entrypoint enables it.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if not _file_logging_state["enabled"]:
            return
        try:
            msg = self.format(record)
            with _file_write_lock:
                with open(_file_logging_state["path"], "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
        except Exception:
            self.handleError(record)


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
        stdout_handler = _CurrentStdoutHandler()
        stdout_handler.setFormatter(JsonFormatter())
        logger.addHandler(stdout_handler)

        file_handler = _OptionalFileHandler()
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
