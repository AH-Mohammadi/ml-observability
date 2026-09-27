import json

from ml_project.reproducibility import (
    get_git_commit,
    metadata_path_for,
    read_model_metadata,
    write_model_metadata,
)


def test_get_git_commit_never_raises_and_returns_string():
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_metadata_path_for_model_joblib():
    path = metadata_path_for("models/baseline.joblib")
    assert str(path) == "models/baseline.meta.json"


def test_write_and_read_model_metadata_roundtrip(tmp_path):
    model_path = tmp_path / "model.joblib"
    meta_path = write_model_metadata(
        model_path,
        model_version="abc123",
        model_type="logistic_regression",
        mlflow_run_id="abc123",
        dataset_path="/data/telco.csv",
        n_train_rows=100,
        n_val_rows=20,
        random_seed=42,
    )
    assert meta_path.exists()
    assert meta_path.name == "model.meta.json"

    loaded = read_model_metadata(model_path)
    assert loaded["model_version"] == "abc123"
    assert loaded["model_type"] == "logistic_regression"
    assert loaded["dataset_path"] == "/data/telco.csv"
    assert loaded["n_train_rows"] == 100
    assert "git_commit" in loaded


def test_read_model_metadata_missing_sidecar_returns_none(tmp_path):
    model_path = tmp_path / "no_metadata_model.joblib"
    result = read_model_metadata(model_path)
    assert result is None


def test_written_metadata_is_valid_json_on_disk(tmp_path):
    model_path = tmp_path / "model.joblib"
    meta_path = write_model_metadata(
        model_path,
        model_version="v2",
        model_type="random_forest",
        mlflow_run_id="v2",
        dataset_path="x.csv",
        n_train_rows=1,
        n_val_rows=1,
        random_seed=1,
    )
    # Should be plain, human-readable JSON, not pickled or otherwise opaque.
    parsed = json.loads(meta_path.read_text())
    assert parsed["model_type"] == "random_forest"
