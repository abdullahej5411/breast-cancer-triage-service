"""
test_api.py -- quick automatic checks for the API.

Run from the project root (after training, so artifacts/model.joblib exists):
  python -m pytest -q
"""

import pytest
from fastapi.testclient import TestClient

from app.counter import RequestCounter
from app.features import EXAMPLE_PAYLOAD
from app.main import app


@pytest.fixture(scope="module")
def client():
    # "with" makes FastAPI run the start-up code (loads the model).
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_the_agreed_shape(client):
    response = client.post("/predict", json=EXAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "malignant_probability", "threshold", "model_version"}
    assert body["label"] in ("malignant", "benign")
    assert 0.0 <= body["malignant_probability"] <= 1.0


def test_example_sample_is_malignant(client):
    # Row 0 of the dataset is a malignant case.
    assert client.post("/predict", json=EXAMPLE_PAYLOAD).json()["label"] == "malignant"


def test_missing_field_gives_422(client):
    payload = dict(EXAMPLE_PAYLOAD)
    payload.pop("mean_radius")
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_value_gives_422(client):
    payload = dict(EXAMPLE_PAYLOAD, mean_radius=-1)
    assert client.post("/predict", json=payload).status_code == 422


def test_text_value_gives_422(client):
    payload = dict(EXAMPLE_PAYLOAD, mean_radius="abc")
    assert client.post("/predict", json=payload).status_code == 422


def test_model_info_has_all_fields(client):
    body = client.get("/model/info").json()
    for key in (
        "model_version",
        "model_family",
        "scikit_learn_version",
        "git_commit",
        "decision_threshold",
        "feature_count",
        "test_metrics",
    ):
        assert key in body
    assert body["feature_count"] == 30


def test_stats_without_redis_is_disabled(client):
    body = client.get("/stats").json()
    assert body == {"predictions_served": None, "redis_status": "disabled"}


def test_predict_still_works_when_redis_is_down(client):
    # Port 1 on localhost has nothing listening, so Redis is "unreachable".
    app.state.counter = RequestCounter("127.0.0.1", 1)
    assert client.post("/predict", json=EXAMPLE_PAYLOAD).status_code == 200
    assert client.get("/stats").json()["redis_status"] == "unreachable"
