# Breast Cancer Triage Service

A small machine learning service that classifies cell-nucleus measurements as
**malignant** or **benign**. It is built with FastAPI, packaged with Docker,
run together with Redis using Docker Compose, and its dataset is versioned with DVC.

> This is an educational exercise for the MLOps course. It is **not** a medical device.

**Docker Hub image:** `docker.io/abdullahej5411/breast-cancer-triage-api` (tags `1.0.0` and `latest`)

---

## 1. What is inside

| Path | What it is |
|---|---|
| `training/prepare_data.py` | Exports the dataset to `data/raw/breast_cancer.csv` and checks it |
| `training/train.py` | Trains two models, picks the best, saves the model files |
| `params.yaml` | Seed (42) and model settings |
| `dvc.yaml`, `dvc.lock` | The DVC training pipeline |
| `data/raw/breast_cancer.csv.dvc` | Small pointer file. The real CSV is stored by DVC |
| `dvc-remote/` | The local DVC remote (committed so teaching staff can run `dvc pull`) |
| `app/` | The FastAPI service |
| `tests/` | Quick automatic tests for the API |
| `Dockerfile` | Production image (multi-stage, non-root, health check) |
| `Dockerfile.naive` | Baseline image used for comparison |
| `docker-compose.yml` | API + Redis, builds the image locally |
| `docker-compose.hub.yml` | API + Redis, uses the published Docker Hub image |
| `.env.example` | Template for the `.env` file (the real `.env` is never committed) |
| `requirements-*.txt` | Pinned Python packages (train / serve / dev) |

## 2. API endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}`. Does not use Redis |
| POST | `/predict` | Takes all 30 features, returns the label and malignant probability |
| GET | `/model/info` | Model version, family, scikit-learn version, git commit, threshold, feature count, test metrics |
| GET | `/stats` | Number of predictions served (counted in Redis) |

Input fields are the 30 dataset feature names with spaces replaced by underscores
(`mean_radius` ... `worst_fractal_dimension`). Every value must be a number `>= 0`.
A missing or invalid field returns HTTP **422**.

Example response from `/predict`:

```json
{"label": "malignant", "malignant_probability": 0.9871, "threshold": 0.5, "model_version": "1.0.0"}
```

In this dataset `target 0 = malignant` and `target 1 = benign`.
Malignant is the **positive** class.

## 3. Configuration (environment variables only)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `artifacts/model.joblib` | Where the model file is |
| `DECISION_THRESHOLD` | `0.5` | A case is malignant when probability >= this value |
| `LOG_LEVEL` | `INFO` | Logging level. Logs go to standard output only |
| `REDIS_HOST` | not set | Request counting is on only when this is set |
| `REDIS_PORT` | `6379` | Redis port |
| `MODEL_VERSION`, `GIT_SHA` | set at build time | Build arguments, shown by `/model/info` |

---

## 4. Commands, from a fresh clone to Docker Hub

Requirements: Git, Python **3.11** (same minor version as the Docker image),
Docker Desktop (WSL 2 backend on Windows) or Docker Engine with the Compose v2 plugin.
No GPU and no cloud account are needed.

Commands work in both PowerShell and bash unless a note says otherwise.

### 4.1 Clone and set up Python

```
git clone https://github.com/abdullahej5411/breast-cancer-triage-service.git
cd breast-cancer-triage-service
python -m venv .venv
```

Activate the virtual environment:

```
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS / WSL
source .venv/bin/activate
```

```
pip install -r requirements-dev.txt
```

### 4.2 Restore the data with DVC

```
dvc pull
```

This creates `data/raw/breast_cancer.csv` (version 2 of the dataset, 539 rows)
from the committed `dvc-remote/` folder.

### 4.3 Train with the DVC pipeline

```
dvc repro
```

This runs `training/train.py` using `params.yaml` (seed 42) and writes
`artifacts/model.joblib`, `artifacts/model_card.json` and `artifacts/metrics.json`.
Running `dvc repro` again without changes does nothing (DVC says the stage did not change).

To train without DVC:

```
python -m training.train --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version 1.0.0
```

### 4.4 Run the API locally (no Docker) and run the tests

```
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000/docs> for the interactive documentation page.
In another terminal:

```
python -m pytest -q
```

### 4.5 Create your `.env` file

```
# Windows PowerShell
Copy-Item .env.example .env
# Linux / macOS / WSL
cp .env.example .env
```

Open `.env` and set at least `DOCKERHUB_USERNAME` and `GIT_SHA`
(get it with `git rev-parse --short HEAD`).

### 4.6 Build the images

The dataset must exist before building (`dvc pull`, step 4.2).

```
# Baseline image
docker build -f Dockerfile.naive -t breast-cancer-triage-api:naive .

# Production image (replace the placeholders)
docker build -t abdullahej5411/breast-cancer-triage-api:1.0.0 --build-arg MODEL_VERSION=1.0.0 --build-arg GIT_SHA=<short-git-sha> .
```

### 4.7 Run the production image on its own (no Redis)

```
docker run -d --name triage -p 8000:8000 abdullahej5411/breast-cancer-triage-api:1.0.0
docker ps
docker logs triage
```

Change a setting without rebuilding:

```
docker run -d --name triage-strict -p 8001:8000 -e DECISION_THRESHOLD=0.3 -e LOG_LEVEL=DEBUG abdullahej5411/breast-cancer-triage-api:1.0.0
```

### 4.8 Start the Compose stack (API + Redis)

```
docker compose up -d --build
docker compose ps
```

Send a few predictions, then check `http://localhost:8000/stats`.

```
# Stop the stack but KEEP the Redis volume (the counter survives)
docker compose down
docker compose up -d

# Remove the stack AND its volumes (the counter starts from zero)
docker compose down -v
```

### 4.9 Push to Docker Hub

```
docker login
docker tag abdullahej5411/breast-cancer-triage-api:1.0.0 abdullahej5411/breast-cancer-triage-api:latest
docker push abdullahej5411/breast-cancer-triage-api:1.0.0
docker push abdullahej5411/breast-cancer-triage-api:latest
```

### 4.10 Run the published image from an empty folder

Put only `docker-compose.hub.yml` and a `.env` file in an empty folder, then:

```
docker compose -f docker-compose.hub.yml up -d
docker compose -f docker-compose.hub.yml ps
```

---

## 5. Design notes

- **Seed 42** is used for the train/test split, cross-validation and every model.
- **Quality gate:** `training/train.py` exits with an error if the malignant
  recall on the test set is below 0.95. A failed gate stops `dvc repro` and also
  stops `docker build`, so a weak model can never be shipped.
- **Model choice:** the model with the highest mean cross-validation malignant
  recall is selected. Malignant cases get a higher class weight (5) in both
  candidates, because missing a malignant case is worse than a false alarm.
  The price is lower precision (more false alarms).
- **Same scikit-learn version in training and serving.** The API refuses to start
  if the versions differ.
- **Model files are not in Git.** They are rebuilt with `dvc repro` (or inside the
  Docker build). Only the dataset pointer, `dvc.yaml`, `dvc.lock` and `params.yaml` are committed.
- **Redis is optional.** If `REDIS_HOST` is not set, or Redis is down, `/predict`
  still works; `/stats` then reports `null` and a `redis_status` of `disabled` or `unreachable`.
- **Pinned versions:** every Python package and every third-party image uses an exact version.
