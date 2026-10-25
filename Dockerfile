# =============================================================================
# Dockerfile  --  the PRODUCTION image (Task 11)
#
# Two stages ("multi-stage build"):
#   Stage 1 "trainer": installs training packages, trains the model.
#   Stage 2 "final"  : installs ONLY serving packages, copies ONLY the model
#                      files from stage 1 and the app code.
#
# Result: the final image has NO dataset, NO pandas, NO training code.
# =============================================================================


# -----------------------------------------------------------------------------
# STAGE 1: train the model
# -----------------------------------------------------------------------------
# A pinned "slim" Python image (exact version tag, never "latest").
FROM python:3.11.9-slim-bookworm AS trainer

# Do not write .pyc files; do not keep pip's download cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy ONLY the requirements file first, then install.
# Docker caches this step, so it re-runs only when requirements-train.txt changes.
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt

# Now copy what training needs: settings, training code, and the dataset.
# The dataset must exist on your computer before you build.
# (Get it with:  dvc pull)
COPY params.yaml .
COPY training/ training/
COPY data/raw/breast_cancer.csv data/raw/breast_cancer.csv

# The model version is given at build time:  --build-arg MODEL_VERSION=1.0.0
ARG MODEL_VERSION=1.0.0

# Train. If test recall is below 0.95, this step fails and the build stops.
# Same seed (42, from params.yaml) as on your laptop.
RUN python -m training.train \
        --data-path data/raw/breast_cancer.csv \
        --output-dir artifacts \
        --model-version "${MODEL_VERSION}"


# -----------------------------------------------------------------------------
# STAGE 2: the small image that is actually shipped
# -----------------------------------------------------------------------------
FROM python:3.11.9-slim-bookworm AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create a normal (non-root) user. The server will run as this user.
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --no-create-home appuser

WORKDIR /app

# Layer order matters for speed (slowest-changing things first):
#   1) serving packages   (change rarely)
#   2) the model files    (change when the data or training code changes)
#   3) the app code       (changes most often)
# So editing app code does NOT reinstall the packages.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Copy ONLY the three output files from the trainer stage (no dataset).
COPY --from=trainer --chown=appuser:appuser /build/artifacts/model.joblib artifacts/model.joblib
COPY --from=trainer --chown=appuser:appuser /build/artifacts/model_card.json artifacts/model_card.json

COPY --chown=appuser:appuser app/ app/

# Build arguments, saved as environment variables (so /model/info can show them)
# and as image labels (so "docker inspect" can show them).
# They are declared late so changing them does not break the cache above.
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown
ENV MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA}
LABEL org.opencontainers.image.title="breast-cancer-triage-api" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}"

# Default settings. You can replace any of them with "docker run -e NAME=value".
# REDIS_HOST is NOT set here, so the image runs fine on its own (no Redis).
ENV MODEL_PATH=artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    REDIS_PORT=6379

# Run as the non-root user from now on.
USER appuser

# Documents the port the server listens on.
EXPOSE 8000

# Check /health every 15 seconds.
# The slim image has no curl, so we use Python (which is already inside).
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# ENTRYPOINT = the fixed program. CMD = default arguments you may replace.
# --host 0.0.0.0 makes the server reachable from OUTSIDE the container.
ENTRYPOINT ["uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
