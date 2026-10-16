# Historical Day 2 Feature Engineering

This document describes the original local/Glue-oriented feature-engineering
path. It is retained for history and is **not** the deployed ETL path. The
deployed path is S3 object creation -> `m2-build-feature-etl-dev` Lambda ->
`m2-build-features-table` DynamoDB. Glue creation was denied by AWS and the
Lambda replacement is the current implementation.

Run from the `ml` directory:

```powershell
python src/feature_engineering.py
```

The script reads `data/travistorrent.csv.gz`, selects only the relevant source
columns, sorts builds by UTC start time, creates leakage-safe historical
repository features, writes `data/training_features.csv`, and writes the EDA
summary to `EDA_REPORT.md`.

The current build is excluded from its own history. `test_coverage_delta` and
`author_recent_failure_rate` are intentionally not included because the source
dataset does not provide valid inputs for them. This script does not train a
model, deploy SageMaker, or modify AWS resources.