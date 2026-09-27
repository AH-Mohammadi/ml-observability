# ml-observability

An end-to-end ML system for predicting customer churn — built not just to
train a model, but to demonstrate operating, debugging, reproducing, and
monitoring one in production. Every stage is runnable locally, and the
project culminates in a reproducible incident: a simulated drift event
that the monitoring stack actually catches.

## 1. Problem

Predict whether a telecom customer will churn, using the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM / Kaggle) — a binary classification problem on ~7,000 customers.

## 2. Architecture

```
CSV (raw data)
    |
load_data()  ─── schema validation
    |
split_data() ─── stratified train/val/test, seeded
    |
Pipeline(preprocess, model)  ─── logistic regression / random forest / gradient boosting
    |                              |
    |                         MLflow (params, metrics, model artifact)
    |
compare.py  ─── validation comparison → select best → evaluate ONCE on test
    |
models/baseline.joblib + models/baseline.meta.json (reproducibility sidecar)
    |
predict.py  ─── data quality checks → prediction → structured logs → metrics
    |
FastAPI (api.py)  ─── POST /predict, GET /metrics
    |
logs/app.jsonl (persisted, searchable)  +  dashboard.py (one-view status)
    |
drift.py  ─── data drift (PSI/KS) + prediction drift (PSI)
    |
scripts/simulate_drift.py  ─── reproduces a detectable incident end-to-end
    |
Docker (Dockerfile, docker-compose.yml)  ─── containerized serving
```

## 3. Dataset

7,043 customers, 21 columns (`customerID`, demographics, service add-ons,
`Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`).
The baseline pipeline uses four features: `tenure`, `MonthlyCharges`,
`Contract`, `PaymentMethod` — enough to be meaningfully predictive and to
support the drift story below, without every service add-on column
adding noise to a demo.

**Drift story:** a pricing change or a new low-cost competitor shifts
customer behavior — more customers move to month-to-month contracts and
electronic-check payment, alongside a price increase. This is a
*simulated* shift (see Limitations) — the dataset is a single snapshot
with no native time axis, so there's no real "before/after" to observe.

## 4. ML pipeline

`data.py` -> `split.py` -> `preprocessing.py` (ColumnTransformer: median
impute + scale for numeric, most-frequent impute + one-hot for
categorical) -> `train.py` (Pipeline, fit on train only) -> `evaluate.py`
(accuracy/precision/recall/F1/ROC-AUC). Strict discipline: preprocessing
is fit once, on train; validation and test only ever see `.transform()`.
`compare.py` trains all three model types, compares them on validation,
selects the best by ROC-AUC, and evaluates that single selection on the
held-out test set exactly once — never used for model selection.

## 5. Experiment tracking

MLflow, backed by a local SQLite db (`mlflow_data/mlflow.db` — chosen
over the plain filesystem store, which recent MLflow versions have put
into maintenance mode). Each training run logs model type, hyperparameters
(via random seed + model params), dataset row counts, validation metrics,
training duration, and the model artifact itself.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_data/mlflow.db
```

## 6. Logging

Structured, single-line JSON via `logging_config.py` — every event
(`training_started`, `prediction_completed`, `prediction_failed`,
`data_quality_warning`, `drift_check_completed`, ...) is a parseable JSON
object with a `timestamp`, `event`, and event-specific fields (`request_id`,
`latency_ms`, `model_version`, etc.). Logs go to stdout always, and
additionally to `logs/app.jsonl` when a CLI entrypoint calls
`enable_file_logging()` — kept opt-in so importing/testing a module never
incurs file I/O.

## 7. Monitoring

`metrics.py` tracks live, in-process counters (request/error/prediction
count, average latency) — this is the **metrics** layer, distinct from
**logs** (discrete per-event records) and **traces** (not needed here —
single process, no multi-service hops). `dashboard.py` combines live
metrics with aggregated, persisted logs and the latest drift status into
one view:

```bash
python -m ml_project.dashboard
python -m ml_project.dashboard --search event=prediction_failed
python -m ml_project.dashboard --search request_id=<id>
```

## 8. Drift detection

`drift.py`: **data drift** (did the input feature distributions change?)
via Population Stability Index for categoricals (`Contract`,
`PaymentMethod`) and a two-sample Kolmogorov-Smirnov test for numerics
(`tenure`, `MonthlyCharges`); **prediction drift** (did the model's output
distribution change?) via PSI on the predicted-class distribution.
Two-tier severity, standard convention: PSI < 0.1 no significant change,
0.1-0.25 moderate (investigate), >= 0.25 major.

**Important:** prediction drift does not by itself mean the model is
failing — it means the population being scored looks different. It's a
signal to investigate, not a verdict. Concept drift (the relationship
between features and the true label changing) is a distinct, harder
problem this project doesn't attempt to detect directly.

**Method limitations:** PSI needs a reasonably sized sample per group to
be stable and is only meaningful for a small, fixed set of categories
(true here); KS detects *that* a distribution changed, not *how* —
summaries include both mean and std so a shift's direction is visible
alongside the statistic.

## 9. Incident simulation

```bash
python scripts/simulate_drift.py
```

Walks through the full story: a normal production batch (no drift) -> a
simulated pricing/competitor shift (the drift story above, applied via
`apply_drift_shift()`) -> drift detected on `Contract`/`PaymentMethod`/
`MonthlyCharges` -> prediction drift on the churn rate -> an investigation
summary naming the affected features. Reproducible with `--seed`.

## 10. Local setup

```bash
git clone <this repo>
cd ml-observability
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and save it as `data/raw/telco_churn.csv`. (Tests don't need this — they
use a small fixture CSV. Training/inference on real data does.)

## 11. Running training

```bash
python -m ml_project.train                          # logistic regression (default)
python -m ml_project.train --model-type random_forest
python -m ml_project.compare                         # compare all 3, select best, eval on test once
```

Produces `models/baseline.joblib` + `models/baseline.meta.json` (model
version, MLflow run id, dataset path, git commit — the reproducibility
sidecar) and logs to MLflow + `logs/app.jsonl`.

## 12. Running inference

```bash
uvicorn ml_project.api:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 55.0, "Contract": "One year", "PaymentMethod": "Mailed check"}'
curl localhost:8000/metrics
```

Or with Docker:

```bash
docker-compose --profile train run train     # train inside the container
docker-compose up                             # serve on :8000
```

## 13. Running monitoring

```bash
python -m ml_project.dashboard
python scripts/simulate_drift.py
```

## 14. Results

*(Fill in after running against the real dataset — this project was
built and verified against synthetic data throughout development, since
the real Telco CSV wasn't available in the build environment. Expect
meaningfully higher ROC-AUC than the placeholder numbers seen during
development once run against real data.)*

## 15. Example incident

See section 9 — `scripts/simulate_drift.py` output is the live
demonstration: healthy baseline -> simulated shift -> drift caught on the
right features -> investigation summary.

## 16. Testing

```bash
pytest -v
```

Unit tests (preprocessing, split, drift math, data quality rules),
integration tests (full training pipeline, full prediction pipeline,
incident simulation end-to-end), and failure tests (missing/invalid
schema, out-of-range values, non-numeric input, missing model file, empty
input). Tests never depend on the real downloaded dataset — fixtures and
synthetic data only, so the suite runs identically for anyone who clones
the repo without having downloaded anything yet.

## 17. Limitations

- The dataset is a single snapshot with no native time axis — all drift
  shown by `simulate_drift.py` is constructed, not observed. A real
  deployment would compare against genuinely time-ordered production data.
- No real ground-truth feedback loop — churn labels for new customers
  aren't available immediately (if ever), so actual model performance in
  production can't be measured directly, only proxied via data/prediction
  drift.
- `model_version` is the MLflow run id, not a semantic version — fine for
  traceability, less fine for human communication ("which model is live"
  requires looking it up, not reading it off a version number).
- The FastAPI service (`api.py`) was built but not runtime-verified during
  development (no network access to install `fastapi` in the build
  environment) — verify it directly before relying on it.
- In-memory metrics reset on process restart; only the persisted log file
  survives across restarts.

## 18. Future improvements

- Real-time drift monitoring on a rolling window of live traffic, rather
  than only via the on-demand simulation script.
- A proper metrics backend (Prometheus + Grafana) once multi-process/
  multi-instance deployment makes in-memory counters insufficient.
- Centralized log aggregation (Loki or similar) if/when logs need to be
  searched across more than one running instance.
- A feedback loop that eventually captures true churn outcomes to measure
  real model performance, not just drift proxies.
