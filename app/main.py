"""
main.py -- the FastAPI service (Task 9 and Task 13).

Read this file from top to bottom. The order is:
  1. Read the settings from environment variables.
  2. Set up logging (everything goes to standard output).
  3. Load the model ONCE when the server starts (and Redis counter setup).
  4. Define the four endpoints:
        GET  /health      -> {"status": "ok"}                (does not use Redis)
        POST /predict     -> label + malignant probability
        GET  /model/info  -> details about the loaded model
        GET  /stats       -> number of predictions served (from Redis)

This is an educational exercise. It is NOT a medical device.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_settings
from app.counter import RequestCounter
from app.model_store import ModelStore
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    StatsResponse,
)

# ----------------------------------------------------------------------------
# Step 1: read the settings (a wrong value stops the server with a clear error)
# ----------------------------------------------------------------------------
settings = load_settings()


# ----------------------------------------------------------------------------
# Step 2: logging. Logs go to standard output ONLY (not to files).
# ----------------------------------------------------------------------------
def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name, logging.INFO)

    # Our own logs (the "app" loggers) go to stdout through the root logger.
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)

    # The web server (uvicorn) writes its start-up lines to stderr by default.
    # We move them to stdout too, and we use the same log level.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        server_logger = logging.getLogger(logger_name)
        server_logger.setLevel(level)
        for server_handler in server_logger.handlers:
            if isinstance(server_handler, logging.StreamHandler):
                server_handler.setStream(sys.stdout)


setup_logging(settings.log_level)
logger = logging.getLogger("app.main")


# ----------------------------------------------------------------------------
# Step 3: start-up. The model is loaded ONE time here, not on every request.
# If the model file is missing, ModelStore raises an error and the server
# stops right away (fail fast).
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting service (git commit: %s)", settings.git_sha)
    app.state.model = ModelStore(settings.model_path)
    app.state.counter = RequestCounter(settings.redis_host, settings.redis_port)
    yield
    logger.info("Service stopped")


app = FastAPI(
    title="Breast Cancer Triage Service",
    description="Educational exercise. Not a medical device.",
    lifespan=lifespan,
)


def current_model_version() -> str:
    """Use the version given at build time. If none, use the one in the model card."""
    return settings.model_version or app.state.model.card["model_version"]


# ----------------------------------------------------------------------------
# Step 4a: GET /health -- never touches Redis, so it is always fast and safe
# ----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


# ----------------------------------------------------------------------------
# Step 4b: POST /predict
# FastAPI + Pydantic already checked the body. If a field is missing or is not
# a number >= 0, the client got a 422 and this function was never called.
# ----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 1. Ask the model for the probability of "malignant".
    probability = app.state.model.malignant_probability(request.model_dump())

    # 2. Compare with the threshold to choose the label.
    label = "malignant" if probability >= settings.decision_threshold else "benign"

    # 3. Count this prediction in Redis (skipped if Redis is off or down).
    app.state.counter.increase()

    logger.info("Prediction made: label=%s", label)
    logger.debug("Malignant probability=%.6f threshold=%s", probability, settings.decision_threshold)

    return {
        "label": label,
        "malignant_probability": round(probability, 4),
        "threshold": settings.decision_threshold,
        "model_version": current_model_version(),
    }


# ----------------------------------------------------------------------------
# Step 4c: GET /model/info
# ----------------------------------------------------------------------------
@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    card = app.state.model.card
    return {
        "model_version": current_model_version(),
        "model_family": card["model_family"],
        "scikit_learn_version": card["scikit_learn_version"],
        "git_commit": settings.git_sha,
        "decision_threshold": settings.decision_threshold,
        "feature_count": len(card["feature_names"]),
        "test_metrics": card["metrics"]["test"],
    }


# ----------------------------------------------------------------------------
# Step 4d: GET /stats -- how many predictions were served (stored in Redis)
# ----------------------------------------------------------------------------
@app.get("/stats", response_model=StatsResponse)
def stats():
    count, status = app.state.counter.read()
    return {"predictions_served": count, "redis_status": status}
