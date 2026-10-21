"""
prepare_data.py  --  Task 5: export the dataset to a CSV file.

What this script does, step by step:
  1. Load the breast cancer dataset that ships with scikit-learn.
  2. Put the 30 feature columns and a "target" column into a table.
  3. Check that there are no empty (null) values.
  4. Check that there are exactly 30 feature columns.
  5. Save the table as data/raw/breast_cancer.csv.

How to run (from the project root folder):
  python -m training.prepare_data

Note about the target column in this dataset:
  target = 0 means "malignant"   (this is the class we want to catch)
  target = 1 means "benign"
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

EXPECTED_FEATURE_COUNT = 30
TARGET_COLUMN = "target"


def main() -> None:
    # ---- Step 0: read the command line argument (where to save the CSV) ----
    parser = argparse.ArgumentParser(description="Export the breast cancer dataset to a CSV file.")
    parser.add_argument(
        "--output-path",
        default="data/raw/breast_cancer.csv",
        help="Where to save the CSV file (default: data/raw/breast_cancer.csv)",
    )
    args = parser.parse_args()

    # ---- Step 1: load the dataset that comes with scikit-learn ----
    dataset = load_breast_cancer()

    # ---- Step 2: build a table. Spaces in column names become underscores ----
    # Example: "mean radius" becomes "mean_radius"
    column_names = [name.replace(" ", "_") for name in dataset.feature_names]
    table = pd.DataFrame(dataset.data, columns=column_names)
    table[TARGET_COLUMN] = dataset.target

    # ---- Step 3: check there are no null (empty) values ----
    null_count = int(table.isnull().sum().sum())
    if null_count != 0:
        raise SystemExit(f"ERROR: found {null_count} null values. The data must have none.")

    # ---- Step 4: check there are exactly 30 feature columns ----
    feature_count = len(table.columns) - 1  # minus the target column
    if feature_count != EXPECTED_FEATURE_COUNT:
        raise SystemExit(
            f"ERROR: expected {EXPECTED_FEATURE_COUNT} feature columns but found {feature_count}."
        )

    # ---- Step 5: save the CSV (create the folder first if it is missing) ----
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    print(f"OK: no null values, {feature_count} features.")
    print(f"Saved {len(table)} rows to {output_path}")


if __name__ == "__main__":
    main()
