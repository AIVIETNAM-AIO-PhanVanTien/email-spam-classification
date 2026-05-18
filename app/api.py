"""
FastAPI service for the spam classifier.

Endpoints:
    GET  /health
    POST /predict             { subject, body }       → { label, spam_probability, threshold }
    POST /admin/reload-model  ─                        reload pkl + vectorizer + scaler from disk

Predictions are appended to PREDICTION_LOG_PATH (default logs/predictions.csv).
The model is loaded once at startup (see src.predict.load_model).
"""
from __future__ import annotations

import csv
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from app.predict import load_model, predict_spam, reload_model  # noqa: E402

PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", str(_BASE / "logs" / "predictions.csv"))
)
_LOG_FIELDS = ["timestamp", "subject", "body", "prediction", "spam_probability"]
_log_lock = threading.Lock()


class EmailInput(BaseModel):
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body text")


class PredictionResponse(BaseModel):
    label: str
    spam_probability: float
    threshold: float


def _append_prediction_log(subject: str, body: str, label: str, proba: float) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not PREDICTION_LOG_PATH.exists()
    with _log_lock, PREDICTION_LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(_LOG_FIELDS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            subject.replace("\n", " ").replace("\r", " "),
            body.replace("\n", " ").replace("\r", " "),
            label,
            f"{proba:.6f}",
        ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        print("[api] Model loaded successfully")
    except FileNotFoundError as e:
        print(f"[api] WARNING: {e}")
    yield


app = FastAPI(
    title="Email Spam Classifier",
    version="2.0.0",
    description=(
        "Classifier + TF-IDF + StandardScaler composed at inference. "
        "Champion = models/best_spam_classifier.pkl; snapshot resolved from models/train.json."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    try:
        state = load_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_type": state["model_type"],
            "winner": state["winner"],
            "snapshot": state["snapshot"],
            "trained_at": state["trained_at"],
            "threshold": state["threshold"],
            "feature_subset": state["feature_subset"],
        }
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}


@app.post("/admin/reload-model")
def reload_model_endpoint() -> dict:
    """Force-reload the classifier + vectorizer + scaler from disk.
    Called by the monthly retrain job after a champion swap."""
    try:
        meta = reload_model()
        return {"status": "reloaded", "metadata": meta}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: EmailInput) -> PredictionResponse:
    if not payload.subject.strip() and not payload.body.strip():
        raise HTTPException(
            status_code=400,
            detail="Subject or body must not be empty.",
        )

    try:
        result = predict_spam(
            payload.subject, payload.body,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _append_prediction_log(
        payload.subject, payload.body,
        result["label"], result["spam_probability"],
    )
    return PredictionResponse(
        label=result["label"],
        spam_probability=result["spam_probability"],
        threshold=result["threshold"],
    )
