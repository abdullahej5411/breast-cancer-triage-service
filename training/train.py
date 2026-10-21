"""
train.py  --  Task 7: train the model.

What this script does, step by step:
  1. Read the settings from params.yaml (seed, model settings, and so on).
  2. Read the CSV file (the dataset tracked by DVC).
  3. Split it 80/20 into a train part and a test part (stratified, seed 42).
  4. Build two candidate models. Each one is a scikit-learn Pipeline,
     so the preprocessing (scaling) travels together with the model.
        - Logistic Regression
        - Random Forest
  5. Test each candidate with 5-fold stratified cross-validation on the TRAIN part.
     We record: malignant recall, accuracy and ROC AUC.
  6. Pick the candidate with the highest mean cross-validation malignant recall.
     (The choice uses ONLY the cross-validation numbers, never the test part.)
  7. Fit every candidate on the train part and measure it on the test part.
     We measure all of them so the report can show a full table,
     but only the selected model is used and gated.
  8. If the selected model's test malignant recall is below 0.95,
     stop with an error (exit code 1).
  9. Otherwise save three files:
        model.joblib      the trained pipeline
        model_card.json   a small description of the model
        metrics.json      all the numbers (DVC reads this file)

How to run (from the project root folder):
  python -m training.train \
      --data-path data/raw/breast_cancer.csv \
      --output-dir artifacts \
      --model-version 1.0.0

Important idea: "malignant" is the POSITIVE class here.
  In the data, target 0 = malignant and target 1 = benign.
  So for recall we use pos_label=0, and the malignant probability
  is the model's probability for class 0.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# In our data, class 0 means malignant. This is the "positive" class.
MALIGNANT_CLASS = 0


# ----------------------------------------------------------------------------
# Step 0: read the command line arguments
# ----------------------------------------------------------------------------
def read_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the breast cancer model.")
    parser.add_argument("--data-path", required=True, help="Path to the CSV file")
    parser.add_argument("--output-dir", required=True, help="Folder where the model files are saved")
    parser.add_argument("--model-version", required=True, help="Version label, for example 1.0.0")
    parser.add_argument(
        "--params-path",
        default="params.yaml",
        help="Path to the settings file (default: params.yaml)",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Step 1: read the settings file
# ----------------------------------------------------------------------------
def read_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ----------------------------------------------------------------------------
# Step 2: read the CSV and split it into features (X) and label (y)
# ----------------------------------------------------------------------------
def load_data(data_path: str, target_column: str):
    table = pd.read_csv(data_path)
    feature_names = [name for name in table.columns if name != target_column]

    # We pass plain numbers (numpy arrays) to the model. This way the saved
    # model does not depend on pandas, and the API does not need pandas.
    X = table[feature_names].to_numpy(dtype=float)
    y = table[target_column].to_numpy(dtype=int)
    return X, y, feature_names


# ----------------------------------------------------------------------------
# Step 4: build the two candidate models
# ----------------------------------------------------------------------------
def build_candidates(params: dict) -> dict:
    seed = params["seed"]
    lr = params["models"]["logistic_regression"]
    rf = params["models"]["random_forest"]

    candidates = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=lr["C"],
                        max_iter=lr["max_iter"],
                        # A missed malignant case is worse than a false alarm,
                        # so class 0 (malignant) gets a bigger weight.
                        class_weight={0: lr["malignant_weight"], 1: 1},
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                # Trees do not need scaling. We keep the same scaler step so both
                # candidates have the same shape. It does no harm.
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=rf["n_estimators"],
                        max_depth=rf["max_depth"],
                        class_weight={0: rf["malignant_weight"], 1: 1},
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    return candidates


# ----------------------------------------------------------------------------
# Step 5: cross-validation of one candidate (on the TRAIN part only)
# ----------------------------------------------------------------------------
def run_cross_validation(pipeline: Pipeline, X_train, y_train, params: dict) -> dict:
    folds = StratifiedKFold(
        n_splits=params["data"]["cv_folds"],
        shuffle=True,
        random_state=params["seed"],
    )
    scoring = {
        # Recall of the malignant class (how many malignant cases we catch)
        "malignant_recall": make_scorer(recall_score, pos_label=MALIGNANT_CLASS),
        "accuracy": "accuracy",
        # ROC AUC gives the same number whether we treat malignant or benign
        # as the positive class, so the built-in scorer is fine.
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring)
    return {
        "malignant_recall": round(float(np.mean(scores["test_malignant_recall"])), 4),
        "accuracy": round(float(np.mean(scores["test_accuracy"])), 4),
        "roc_auc": round(float(np.mean(scores["test_roc_auc"])), 4),
    }


# ----------------------------------------------------------------------------
# Step 7: measure the chosen model on the held-out test part
# ----------------------------------------------------------------------------
def evaluate_on_test(pipeline: Pipeline, X_test, y_test, threshold: float) -> dict:
    # Column order of predict_proba follows pipeline.classes_ (here [0, 1]).
    malignant_column = list(pipeline.classes_).index(MALIGNANT_CLASS)
    malignant_probability = pipeline.predict_proba(X_test)[:, malignant_column]

    # A case is called malignant when probability >= threshold.
    predicted_malignant = malignant_probability >= threshold
    actual_malignant = y_test == MALIGNANT_CLASS

    return {
        "malignant_recall": round(float(recall_score(actual_malignant, predicted_malignant)), 4),
        "malignant_precision": round(
            float(precision_score(actual_malignant, predicted_malignant)), 4
        ),
        "accuracy": round(float(accuracy_score(actual_malignant, predicted_malignant)), 4),
        "roc_auc": round(float(roc_auc_score(actual_malignant, malignant_probability)), 4),
    }


def main() -> None:
    args = read_arguments()
    params = read_params(args.params_path)
    seed = params["seed"]
    threshold = params["evaluation"]["decision_threshold"]
    min_recall = params["evaluation"]["min_malignant_recall"]

    # Set the seed everywhere so that every run gives the same result.
    random.seed(seed)
    np.random.seed(seed)

    # ---- Step 2: read the data ----
    X, y, feature_names = load_data(args.data_path, params["data"]["target_column"])
    print(f"Read {len(X)} rows and {len(feature_names)} features from {args.data_path}")

    # ---- Step 3: stratified 80/20 split with random_state=42 ----
    # "stratify=y" keeps the same malignant/benign ratio in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        stratify=y,
        random_state=seed,
    )
    print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")

    # ---- Steps 4 and 5: build candidates and cross-validate each one ----
    candidates = build_candidates(params)
    cv_results = {}
    for name, pipeline in candidates.items():
        cv_results[name] = run_cross_validation(pipeline, X_train, y_train, params)
        print(f"Cross-validation  {name}: {cv_results[name]}")

    # ---- Step 6: pick the candidate with the highest mean CV malignant recall ----
    # Only cross-validation numbers are used for this choice.
    selected_name = max(cv_results, key=lambda name: cv_results[name]["malignant_recall"])
    selected_pipeline = candidates[selected_name]
    print(f"Selected model (highest CV malignant recall): {selected_name}")

    # ---- Step 7: fit every candidate on the train part, measure on the test part ----
    test_results_all = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        test_results_all[name] = evaluate_on_test(pipeline, X_test, y_test, threshold)
        print(f"Test set results  {name}: {test_results_all[name]}")
    test_results = test_results_all[selected_name]

    # ---- Step 8: the quality gate ----
    if test_results["malignant_recall"] < min_recall:
        print(
            f"ERROR: test malignant recall {test_results['malignant_recall']} "
            f"is below the required {min_recall}. No model was saved.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Step 9: save the three files ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_pipeline, output_dir / "model.joblib")

    model_card = {
        "model_version": args.model_version,
        "model_family": selected_name,
        "feature_names": feature_names,  # the exact order the model expects
        "metrics": {"test": test_results, "cross_validation": cv_results[selected_name]},
        "seed": seed,
        "scikit_learn_version": sklearn.__version__,
        "decision_threshold": threshold,
    }
    with open(output_dir / "model_card.json", "w", encoding="utf-8") as file:
        json.dump(model_card, file, indent=2)

    all_metrics = {
        "selected_model": selected_name,
        "cross_validation": cv_results,       # every candidate
        "test_all_candidates": test_results_all,  # every candidate
        "test": test_results,                 # the selected model only
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    print(f"Saved model.joblib, model_card.json and metrics.json to {output_dir}")


if __name__ == "__main__":
    main()
