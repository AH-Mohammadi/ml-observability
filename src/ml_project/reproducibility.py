"""Reproducibility: trace a prediction back to the run/data/code that
produced its model.

A prediction should be traceable back to:
    request -> model version -> training run (MLflow) -> dataset version -> code version (git commit)

Implemented as a small JSON "sidecar" file saved next to the model
artifact (e.g. models/baseline.meta.json alongside models/baseline.joblib),
rather than baking metadata into the model file itself — keeps the model
file a plain joblib-loadable sklearn Pipeline, and the metadata
human-readable/greppable on its own.
"""

import json
import subprocess
from pathlib import Path


def get_git_commit() -> str:
    """Best-effort short git commit hash for the current checkout.

    Returns "unknown" (never raises) if git isn't available, this isn't a
    git repository, or the command fails for any reason — reproducibility
    metadata should degrade gracefully, not break training.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def metadata_path_for(model_path: Path | str) -> Path:
    """The sidecar metadata path for a given model artifact path.

    models/baseline.joblib -> models/baseline.meta.json
    """
    path = Path(model_path)
    return path.with_suffix("").with_suffix(".meta.json")


def write_model_metadata(
    model_path: Path | str,
    *,
    model_version: str,
    model_type: str,
    mlflow_run_id: str,
    dataset_path: str,
    n_train_rows: int,
    n_val_rows: int,
    random_seed: int,
) -> Path:
    """Write the metadata sidecar for a freshly trained model. Returns the path written."""
    metadata = {
        "model_version": model_version,
        "model_type": model_type,
        "mlflow_run_id": mlflow_run_id,
        "dataset_path": dataset_path,
        "n_train_rows": n_train_rows,
        "n_val_rows": n_val_rows,
        "random_seed": random_seed,
        "git_commit": get_git_commit(),
    }
    meta_path = metadata_path_for(model_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2))
    return meta_path


def read_model_metadata(model_path: Path | str) -> dict | None:
    """Read the metadata sidecar for a model, if it exists.

    Returns None (never raises) if the sidecar is missing — lets callers
    fall back to a placeholder rather than crashing on an older model
    saved before this iteration existed.
    """
    meta_path = metadata_path_for(model_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
