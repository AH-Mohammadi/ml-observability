"""FastAPI inference service.

Run with:
    uvicorn ml_project.api:app --reload

Endpoints:
    POST /predict  -> {prediction, probability, model_version}
    GET  /metrics  -> {request_count, error_count, prediction_count, avg_latency_ms}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_project.metrics import metrics
from ml_project.predict import PredictionError, predict

app = FastAPI(title="Telco Churn Inference API")


class ChurnFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    Contract: str
    PaymentMethod: str


@app.post("/predict")
def predict_endpoint(features: ChurnFeatures):
    try:
        return predict(features.model_dump())
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()
