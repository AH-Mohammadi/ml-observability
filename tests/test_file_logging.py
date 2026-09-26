import json

import pytest

from ml_project.logging_config import disable_file_logging, enable_file_logging, get_logger


@pytest.fixture(autouse=True)
def reset_file_logging():
    disable_file_logging()
    yield
    disable_file_logging()


def test_file_logging_disabled_by_default_no_file_created(tmp_path):
    log_path = tmp_path / "should_not_exist.jsonl"
    logger = get_logger("ml_project.test_file_logging_off")
    logger.info("something_happened", extra={"foo": "bar"})
    assert not log_path.exists()


def test_enable_file_logging_writes_jsonl(tmp_path):
    log_path = tmp_path / "app.jsonl"
    enable_file_logging(log_path)
    logger = get_logger("ml_project.test_file_logging_on")
    logger.info("something_happened", extra={"request_id": "req-1"})

    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == "something_happened"
    assert parsed["request_id"] == "req-1"


def test_disable_file_logging_stops_writes(tmp_path):
    log_path = tmp_path / "app.jsonl"
    enable_file_logging(log_path)
    logger = get_logger("ml_project.test_file_logging_toggle")
    logger.info("event_one")
    disable_file_logging()
    logger.info("event_two")

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1  # only event_one was written


def test_file_logging_appends_across_multiple_calls(tmp_path):
    log_path = tmp_path / "app.jsonl"
    enable_file_logging(log_path)
    logger = get_logger("ml_project.test_file_logging_append")
    logger.info("event_a")
    logger.info("event_b")

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
