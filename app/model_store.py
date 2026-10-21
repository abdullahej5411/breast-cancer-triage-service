"""
model_store.py -- loads the trained model ONCE and makes predictions.

The model file (model.joblib) and the model card (model_card.json) sit in the
same folder. The card tells us the model family, the feature order, the metrics
and the scikit-learn version that was used for training.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import sklearn

from app.features import FEATURE_NAMES

logger = logging.getLogger("app.model")

# In our data, class 0 means malignant.
MALIGNANT_CLASS = 0


class ModelStore:
    def __init__(self, model_path: str):
        model_file = Path(model_path)
        card_file = model_file.parent / "model_card.json"

        # ---- Fail fast with a clear message if a file is missing ----
        for required_file in (model_file, card_file):
            if not required_file.is_file():
                message = f"Model artifact not found: {required_file}"
                logger.critical(message)
                raise RuntimeError(message)

        # ---- Read the model card ----
        with open(card_file, "r", encoding="utf-8") as file:
            self.card = json.load(file)

        # ---- The model must be loaded with the SAME scikit-learn that trained it ----
        trained_with = self.card["scikit_learn_version"]
        if trained_with != sklearn.__version__:
            message = (
                f"scikit-learn version mismatch: model was trained with {trained_with} "
                f"but the server has {sklearn.__version__}"
            )
            logger.critical(message)
            raise RuntimeError(message)

        # ---- The model must expect the same 30 features, in the same order ----
        if self.card["feature_names"] != FEATURE_NAMES:
            message = "The feature names in model_card.json do not match app/features.py"
            logger.critical(message)
            raise RuntimeError(message)

        # ---- Load the model file ----
        self.pipeline = joblib.load(model_file)
        self.malignant_column = list(self.pipeline.classes_).index(MALIGNANT_CLASS)
        logger.info(
            "Model loaded: family=%s version=%s",
            self.card["model_family"],
            self.card["model_version"],
        )

    def malignant_probability(self, values: dict) -> float:
        """Give the 30 values as a dict. Returns the probability of malignant."""
        # Put the values in the exact order the model expects.
        row = [[values[name] for name in FEATURE_NAMES]]
        probabilities = self.pipeline.predict_proba(np.array(row, dtype=float))
        return float(probabilities[0, self.malignant_column])
