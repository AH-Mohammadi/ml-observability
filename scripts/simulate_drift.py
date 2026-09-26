#!/usr/bin/env python
"""Entrypoint: python scripts/simulate_drift.py

Thin wrapper — the actual logic lives in ml_project.simulate_drift so it's
importable and testable without going through argparse/sys.path tricks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_project.simulate_drift import main  # noqa: E402

if __name__ == "__main__":
    main()
