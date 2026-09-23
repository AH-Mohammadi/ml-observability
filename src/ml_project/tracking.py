"""Shared MLflow tracking configuration.

Uses a local SQLite database as the tracking backend by default, rather
than the plain filesystem store (`./mlruns`) — recent MLflow versions have
put the filesystem backend into maintenance mode and will eventually
require a database backend anyway. SQLite is a reasonable default for a
single-machine local project; a real deployment would point
MLFLOW_TRACKING_URI at a proper tracking server instead.

Respects an existing MLFLOW_TRACKING_URI (set by the environment, or by a
test fixture) rather than overriding it, so tests can still point at an
isolated temp database.
"""

import os
from pathlib import Path

import mlflow

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "mlflow.db"


def configure_default_tracking() -> None:
    """Set the MLflow tracking URI to a local SQLite db, unless already set."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    if mlflow.get_tracking_uri().startswith("sqlite:"):
        return
    mlflow.set_tracking_uri(f"sqlite:///{DEFAULT_DB_PATH}")
