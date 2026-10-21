"""
config.py -- reads ALL settings from environment variables.

Why? So the exact same Docker image can run anywhere.
We change behaviour by changing environment variables, never the code.

Variable          Default                   Meaning
----------------  ------------------------  ---------------------------------------
MODEL_PATH        artifacts/model.joblib    where the trained model file is
DECISION_THRESHOLD 0.5                      malignant if probability >= this value
LOG_LEVEL         INFO                      how much to log (DEBUG, INFO, WARNING...)
REDIS_HOST        (not set)                 if not set, request counting is OFF
REDIS_PORT        6379                      Redis port
MODEL_VERSION     (read from model card)    set at build time by Docker
GIT_SHA           unknown                   set at build time by Docker
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    model_path: str
    decision_threshold: float
    log_level: str
    redis_host: Optional[str]
    redis_port: int
    model_version: Optional[str]
    git_sha: str


def load_settings() -> Settings:
    """Read the environment variables and return them as one Settings object."""

    # --- DECISION_THRESHOLD must be a number between 0 and 1 ---
    raw_threshold = os.environ.get("DECISION_THRESHOLD", "0.5")
    try:
        threshold = float(raw_threshold)
    except ValueError:
        raise RuntimeError(f"DECISION_THRESHOLD must be a number, but got '{raw_threshold}'")
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(f"DECISION_THRESHOLD must be between 0 and 1, but got {threshold}")

    # --- REDIS_HOST: an empty value counts as "not set" ---
    redis_host = os.environ.get("REDIS_HOST", "").strip() or None

    # --- REDIS_PORT must be a whole number ---
    raw_port = os.environ.get("REDIS_PORT", "6379")
    try:
        redis_port = int(raw_port)
    except ValueError:
        raise RuntimeError(f"REDIS_PORT must be a whole number, but got '{raw_port}'")

    return Settings(
        model_path=os.environ.get("MODEL_PATH", "artifacts/model.joblib"),
        decision_threshold=threshold,
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        redis_host=redis_host,
        redis_port=redis_port,
        model_version=os.environ.get("MODEL_VERSION", "").strip() or None,
        git_sha=os.environ.get("GIT_SHA", "").strip() or "unknown",
    )
