import json
import logging

from ml_project.logging_config import JsonFormatter, get_logger


def test_json_formatter_produces_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="something_happened", args=(), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)  # raises if not valid JSON
    assert parsed["event"] == "something_happened"


def test_json_formatter_includes_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="training_started", args=(), exc_info=None,
    )
    record.run_id = "abc123"
    record.model_type = "logistic_regression"

    parsed = json.loads(formatter.format(record))
    assert parsed["run_id"] == "abc123"
    assert parsed["model_type"] == "logistic_regression"


def test_json_formatter_includes_required_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="ml_project.train", level=logging.INFO, pathname="", lineno=0,
        msg="training_completed", args=(), exc_info=None,
    )
    parsed = json.loads(formatter.format(record))
    for field in ("timestamp", "level", "logger", "event"):
        assert field in parsed


def test_get_logger_does_not_duplicate_handlers():
    logger1 = get_logger("ml_project.test_dedupe")
    logger2 = get_logger("ml_project.test_dedupe")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1


def test_logger_output_is_parseable_json(capsys):
    logger = get_logger("ml_project.test_output")
    logger.info("prediction_completed", extra={"request_id": "req-1", "latency_ms": 12})

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["event"] == "prediction_completed"
    assert parsed["request_id"] == "req-1"
    assert parsed["latency_ms"] == 12
