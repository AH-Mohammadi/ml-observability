"""Data loading for the Telco Customer Churn dataset.

Dataset: IBM / Kaggle "Telco Customer Churn"
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The raw CSV is not bundled with this repo. Download it and place it at
`data/raw/telco_churn.csv` before running `load_data()`.
"""

from pathlib import Path

import pandas as pd

# Column expected to exist in any valid Telco Churn CSV (raw or fixture).
TARGET_COLUMN = "Churn"

# A minimal set of columns we rely on existing. Kept small and specific
# rather than listing all 21 raw columns, so schema checks stay meaningful
# even if the raw file's column set evolves slightly.
REQUIRED_COLUMNS = [
    "customerID",
    "tenure",
    "Contract",
    "MonthlyCharges",
    "PaymentMethod",
    TARGET_COLUMN,
]

DEFAULT_RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_churn.csv"


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset from a local CSV.

    Parameters
    ----------
    path:
        Optional path to a CSV file. Defaults to ``data/raw/telco_churn.csv``
        at the project root.

    Returns
    -------
    pd.DataFrame
        The raw churn dataset, unmodified.

    Raises
    ------
    FileNotFoundError
        If no file exists at the resolved path. The message explains where
        to download the dataset from.
    ValueError
        If the file loads but is missing expected columns, or is empty.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_RAW_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"No dataset found at {resolved_path}.\n"
            "Download the Telco Customer Churn CSV from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            f"and save it to {resolved_path}."
        )

    df = pd.read_csv(resolved_path)

    if df.empty:
        raise ValueError(f"Dataset at {resolved_path} loaded but contains zero rows.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at {resolved_path} is missing expected columns: {missing}"
        )

    return df
