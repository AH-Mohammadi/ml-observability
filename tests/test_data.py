from pathlib import Path

import pytest

from ml_project.data import TARGET_COLUMN, load_data

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_telco.csv"


def test_data_loads():
    df = load_data(FIXTURE_PATH)
    assert df is not None


def test_dataset_not_empty():
    df = load_data(FIXTURE_PATH)
    assert len(df) > 0


def test_expected_columns_exist():
    df = load_data(FIXTURE_PATH)
    for col in ["customerID", "tenure", "Contract", "MonthlyCharges", "PaymentMethod"]:
        assert col in df.columns


def test_target_column_exists():
    df = load_data(FIXTURE_PATH)
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()) <= {"Yes", "No"}


def test_missing_file_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="No dataset found"):
        load_data(Path("does/not/exist.csv"))


def test_missing_required_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("customerID,tenure\n1,5\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_data(bad_csv)
