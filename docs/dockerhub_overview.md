# breast-cancer-triage-api

A small machine learning web service that classifies breast cell-nucleus measurements
as **malignant** or **benign**. Built with FastAPI and scikit-learn.

> Educational exercise for an MLOps course. **Not a medical device.**

## What the service does

You send 30 numeric measurements to `POST /predict`. The service answers with a label,
the malignant probability, the decision threshold and the model version.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check. Returns `{"status": "ok"}` |
| POST | `/predict` | Prediction for one record (30 features, each a number >= 0) |
| GET | `/model/info` | Model version, family, scikit-learn version, git commit, threshold, test metrics |
| GET | `/stats` | Number of predictions served (needs Redis) |

Interactive docs: `http://localhost:8000/docs`

## Tags

| Tag | Meaning |
|---|---|
| `1.0.0` | First release (pinned, recommended) |
| `latest` | Points to the same image as `1.0.0` |

## Run with `docker run` (no Redis)

```
docker run -d --name triage -p 8000:8000 <dockerhub-username>/breast-cancer-triage-api:1.0.0
```

Then open <http://localhost:8000/docs>.

## Run with Docker Compose (API + Redis)

Create a `.env` file:

```
DOCKERHUB_USERNAME=<dockerhub-username>
IMAGE_TAG=1.0.0
HOST_PORT=8000
DECISION_THRESHOLD=0.5
LOG_LEVEL=INFO
```

Save `docker-compose.hub.yml` from the GitHub repository next to it and run:

```
docker compose -f docker-compose.hub.yml up -d
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DECISION_THRESHOLD` | `0.5` | Malignant when probability >= this value |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` (logs go to stdout) |
| `MODEL_PATH` | `artifacts/model.joblib` | Location of the model file inside the image |
| `REDIS_HOST` | not set | Request counting is on only when set |
| `REDIS_PORT` | `6379` | Redis port |

## Port

The server listens on port **8000** inside the container (exposed in the image).
The image runs as a non-root user and has a built-in health check.

## Source code

GitHub: <https://github.com/<your-github-username>/breast-cancer-triage-service>
