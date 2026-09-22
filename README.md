# ml-observability

A small, incrementally-built ML system: train a churn model, then make it
observable — structured logs, monitoring, drift detection, and a
reproducible incident simulation.

**Problem:** predict customer churn on the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM / Kaggle).

**Drift story:** a pricing change or a new low-cost competitor shifts
customer behavior over time — month-to-month customers on higher
`MonthlyCharges` plans churn more, and more customers move to
`PaymentMethod = Electronic check` (a known high-churn signal in this
dataset). This is a *simulated* shift for demonstration purposes — the
source dataset is a single snapshot with no native time axis.

## Status: Iteration 0 — Skeleton + Data Loading

This iteration adds:
- a minimal installable Python package (`ml_project`)
- a `load_data()` function that reads the raw CSV and validates its schema
- tests using a small fixture CSV (not the full dataset) so tests don't
  depend on downloading anything

## Setup

```bash
cd ml-observability
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Get the real dataset (needed for later iterations, not for tests)

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Save it as `data/raw/telco_churn.csv`

Tests do **not** require this — they use `tests/fixtures/sample_telco.csv`,
a 10-row fixture with the same schema.

## Run

```bash
pytest
```

## Expected result

All tests pass:

```
tests/test_data.py ......                                    [ 85%]
tests/test_pipeline.py .                                     [100%]

====== 7 passed in 0.XXs ======
```

## Verification

```bash
python -c "from ml_project.data import load_data; df = load_data('tests/fixtures/sample_telco.csv'); print(df.shape); print(df['Churn'].value_counts())"
```

Expected: `(10, 21)` and a `Churn` value count of `No: 6, Yes: 4`.

## Git checkpoint

```bash
git init
git add .
git commit -m "Iteration 0: project skeleton + data loading with schema validation"
```

## What we have now

```
Data (CSV, local file) → load_data() → validated DataFrame
```

No model, no split, no preprocessing yet — just a package that installs,
a loader that fails loudly and clearly on a missing/malformed file, and
tests that prove it.

## Next step

Iteration 1: train/validation/test split with a reproducible seed and
leakage-safety tests (fit preprocessing on train only).

## Limitations

- The drift simulated in later iterations is constructed, not observed —
  the raw dataset has no timestamp column.
