"""Prepare the Day 4 temporal splits for SageMaker built-in XGBoost."""

import json
from pathlib import Path

import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ML_ROOT / "data" / "splits"
OUTPUT_DIR = ML_ROOT / "data" / "sagemaker_xgboost"

FEATURE_COLUMNS = [
    "files_changed",
    "lines_changed",
    "repo_recent_failure_rate",
    "time_of_day_bucket",
    "day_of_week",
    "prior_failures_last_5_builds",
]
LABEL_COLUMN = "label"
OUTPUT_COLUMNS = [LABEL_COLUMN] + FEATURE_COLUMNS
CATEGORY_MAPPINGS = {
    "time_of_day_bucket": {
        "night": 0,
        "morning": 1,
        "afternoon": 2,
        "evening": 3,
    },
    "day_of_week": {
        "Mon": 0,
        "Tue": 1,
        "Wed": 2,
        "Thu": 3,
        "Fri": 4,
        "Sat": 5,
        "Sun": 6,
    },
}


def transform_chunk(frame):
    expected = FEATURE_COLUMNS + [LABEL_COLUMN]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected input columns: {list(frame.columns)}")
    if frame.isna().any().any():
        raise ValueError("Input contains missing values")

    result = frame.copy()
    for column, mapping in CATEGORY_MAPPINGS.items():
        unknown = set(result[column].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unknown values in {column}: {sorted(unknown)}")
        result[column] = result[column].map(mapping).astype("int8")

    result[LABEL_COLUMN] = pd.to_numeric(result[LABEL_COLUMN], errors="raise").astype("int8")
    for column in [
        "files_changed",
        "lines_changed",
        "repo_recent_failure_rate",
        "prior_failures_last_5_builds",
    ]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.isna().any().any():
        raise ValueError("Transformation produced missing values")
    return result[OUTPUT_COLUMNS]


def prepare_split(name, chunksize=250_000):
    input_path = INPUT_DIR / f"{name}.csv"
    output_path = OUTPUT_DIR / f"{name}.csv"
    row_count = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            transformed = transform_chunk(chunk)
            transformed.to_csv(output, index=False, header=False)
            row_count += len(transformed)
    return row_count


def main():
    counts = {name: prepare_split(name) for name in ("train", "validation", "test")}
    manifest = {
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "format": "CSV without header; label is first column for SageMaker XGBoost",
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "output_columns": OUTPUT_COLUMNS,
        "categorical_encoding": "fixed deterministic integer encoding",
        "category_mappings": CATEGORY_MAPPINGS,
        "shuffling": False,
        "row_counts": counts,
        "missing_values": False,
        "dropped_rows": 0,
        "limitations": [
            "Integer encoding preserves six logical model features and keeps files compact.",
            "The integer order is deterministic but categories are nominal; evaluate one-hot encoding if model quality is inadequate.",
            "Chronological membership is inherited unchanged from data/splits/manifest.json.",
        ],
    }
    manifest_path = OUTPUT_DIR / "preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()