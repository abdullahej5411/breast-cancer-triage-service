"""
schemas.py -- the shapes of the data going into and out of the API.

Pydantic checks every request body. If something is wrong (a missing field,
a negative number, text instead of a number) FastAPI answers with HTTP 422.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.features import EXAMPLE_PAYLOAD, FEATURE_NAMES

# A feature value must be a real number that is greater than or equal to 0.
# allow_inf_nan=False also rejects NaN and infinity.
NonNegativeNumber = Annotated[float, Field(ge=0, allow_inf_nan=False)]

# Build the request model from the list of 30 names.
# Every field is required ("..." means required), so a missing field gives 422.
PredictionRequest = create_model(
    "PredictionRequest",
    __config__=ConfigDict(json_schema_extra={"example": EXAMPLE_PAYLOAD}),
    **{name: (NonNegativeNumber, ...) for name in FEATURE_NAMES},
)


class PredictionResponse(BaseModel):
    label: Literal["malignant", "benign"]
    malignant_probability: float
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_family: str
    scikit_learn_version: str
    git_commit: str
    decision_threshold: float
    feature_count: int
    test_metrics: dict


class StatsResponse(BaseModel):
    # None means "we could not count" (Redis is off or not reachable).
    predictions_served: Optional[int]
    redis_status: Literal["ok", "disabled", "unreachable"]
