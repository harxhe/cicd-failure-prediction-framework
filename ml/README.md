# M2 — ML & Prediction Layer

## Current Workspace and Deployment Status

This workspace supports the CI/CD build failure prediction project. The ML
backup dataset was used for EDA and model development because the real project
data initially contained only eight CodeBuild records, which was too small for
meaningful training.

- **AWS region:** `ap-south-1` (Mumbai)
- **S3 bucket:** `cicdfp-data-lake-dev`
- **Real project data:** `s3://cicdfp-data-lake-dev/raw/cicdfp-sample-app-build/`
- **Backup dataset:** `travistorrent.csv.gz`
- **ETL:** `m2-build-feature-etl-dev` Lambda, triggered by S3 object creation
      events under the raw build prefix
- **Feature storage:** `m2-build-features-table` DynamoDB table
- **Analytics:** Athena table `cicdfp_m2.m2_training_features`
- **Inference:** SageMaker endpoint `m2-xgboost-endpoint-v2-dev` was reported as
      `InService`; live status is not independently verifiable with current
      AWS CLI permissions
- **Validation:** manual Lambda tests, S3-triggered testing, DynamoDB checks,
      and the current local test suite have passed (`97 passed`)

Place `travistorrent.csv.gz` under `ml/data/` when available. Run
`python src/prepare_travistorrent.py` to inspect its columns and create a
Day 1-ready CSV containing only the source fields needed for EDA. The script
does not calculate project features or train a model.

See [DATA_MAPPING.md](DATA_MAPPING.md) for the source-to-project feature map.

## Role

M2 is responsible for the **ML & Prediction** layer of the AI-Based CI/CD Failure Prediction Framework. M2 owns **Athena**, **DynamoDB**, and **SageMaker** (specs.md §1).

## Architecture

```
M1 raw build events (S3 raw/)
        │
        ▼
  Lambda ETL (m2-build-feature-etl-dev)
        │
        ▼
  Six processed model features
        │
        ├──▶ DynamoDB (real-time feature lookup)
        ├──▶ Athena (cicdfp_m2.m2_training_features)
        └──▶ SageMaker Training
                 │
                 ▼
           Model Artifact
                 │
                 ▼
           SageMaker Endpoint (m2-xgboost-endpoint-v2-dev)
                 │
                 ▼
           M3 Step Functions / API Gateway
```

## Data Flow

| Step | Input | Output | Owner |
|------|-------|--------|-------|
| Raw events → S3 | CodeBuild events | `s3://cicdfp-data-lake-dev/raw/{repo}/{build_id}.json` | M1 |
| Raw → Features | S3 event → Lambda | Six model features | M2 |
| Features → Athena | Processed feature records | `cicdfp_m2.m2_training_features` | M2 |
| Features → DynamoDB | Lambda feature records | `m2-build-features-table` (recent history lookup) | M2 |
| Training | Processed features | Model artifact | M2 |
| Inference | Feature vector | `{ risk_score, risk_level }` | M2 endpoint → M3 consumer |

## Raw Schema (M1 → M2, specs.md §4)

| Field | Type | Notes |
|-------|------|-------|
| `build_id` | string | Unique, from CodeBuild |
| `repo` | string | Repository name |
| `branch` | string | |
| `commit_sha` | string | |
| `author` | string | Nullable |
| `timestamp` | ISO-8601 | Build start time |
| `files_changed` | integer | Nullable |
| `lines_added` | integer | Nullable |
| `lines_deleted` | integer | Nullable |
| `test_count` | integer | Nullable |
| `build_duration_seconds` | integer | Null if still running |
| `status` | string | `SUCCEEDED \| FAILED \| IN_PROGRESS` |
| `failed_stage` | string | Nullable |

## Processed Feature Schema (specs.md §5)

The deployed model intentionally uses six numeric features:
`files_changed`, `lines_changed`, `repo_recent_failure_rate`,
`time_of_day_bucket`, `day_of_week`, and `prior_failures_last_5_builds`.
The shared schema below still documents eight features; `test_coverage_delta`
and `author_recent_failure_rate` are unavailable and are not sent to the
deployed endpoint. This difference requires team approval and is not changed
locally here.

| Feature | Type | Source | Notes |
|---------|------|--------|-------|
| `build_id` | string | Passthrough | |
| `repo` | string | Passthrough | |
| `files_changed` | int | Raw field | NaN if null |
| `lines_changed` | int | `lines_added + lines_deleted` | NaN if either null |
| `test_coverage_delta` | float | **PLACEHOLDER** | See limitation below |
| `author_recent_failure_rate` | float 0-1 | Trailing 30 days, cross-repo | |
| `repo_recent_failure_rate` | float 0-1 | Trailing 30 days | |
| `time_of_day_bucket` | string | Timestamp hour | `morning\|afternoon\|evening\|night` |
| `day_of_week` | string | Timestamp | `Mon\|Tue\|...\|Sun` |
| `prior_failures_last_5_builds` | int 0-5 | Previous 5 repo builds | |
| `label` | int/null | Status mapping | 0=SUCCEEDED, 1=FAILED, null=IN_PROGRESS |

## Feature Engineering Details

### files_changed
Direct passthrough from raw. Returns NaN when null.

### lines_changed
`lines_added + lines_deleted`. Returns NaN if either source is null.

### test_coverage_delta ⚠️
**PLACEHOLDER**: The raw schema provides `test_count` (integer) but NOT test coverage percentage. This feature requires coverage percentage readings that are unavailable. Currently returns `0.0` for all records. **This is NOT real data.** Dependency: M1 must add coverage fields.

### author_recent_failure_rate
Ratio of FAILED builds to total completed (SUCCEEDED+FAILED) builds by the same author in the 30 days before the current build. Cross-repo scope. IN_PROGRESS excluded. Returns 0.0 for unknown/null authors or no history.

### repo_recent_failure_rate
Same as above but scoped to the repository instead of the author.

### time_of_day_bucket
Derived from the timestamp hour. Boundaries (ASSUMPTION — not defined in specs.md):
- `night`: 00:00–05:59
- `morning`: 06:00–11:59
- `afternoon`: 12:00–17:59
- `evening`: 18:00–23:59

### day_of_week
Day name from timestamp: `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, `Sun`.

### prior_failures_last_5_builds
Count of FAILED builds among the 5 most recent completed builds for the same repo before the current build. IN_PROGRESS excluded. Returns 0 when fewer than 5 builds exist.

### label
`SUCCEEDED → 0`, `FAILED → 1`, `IN_PROGRESS → None`.

## Data Leakage Prevention

- **Current build excluded**: The current build's outcome is never used to calculate its own features.
- **Chronological ordering**: All historical features use only builds with timestamps strictly before the current build.
- **30-day window**: Failure rates use a sliding window relative to each build's own timestamp.

## How to Run Locally

```bash
cd "Cloud Project/ml"

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run EDA script (VS Code: open and use "Run Cell")
python notebooks/m2_eda_pipeline.py
```

## How to Connect Real M1 Data

1. Confirm the Lambda receives the S3 source:
      - Bucket: `cicdfp-data-lake-dev`
   - Prefix: `raw/{repo}/{build_id}.json`
2. The Lambda reads each JSON object and derives the six deployed model features.
3. Confirm the feature item is written to DynamoDB.
4. Query `cicdfp_m2.m2_training_features` for analytics and evaluation data.

The deployed Lambda writes processed records to DynamoDB only. It does not write
processed Parquet or JSONL files to S3. The Athena table is reported as
`cicdfp_m2.m2_training_features`; its DDL and physical source location must be
verified in AWS.

The local SageMaker CSV format is headerless with this order:

```text
label,files_changed,lines_changed,repo_recent_failure_rate,time_of_day_bucket,day_of_week,prior_failures_last_5_builds
```

The deployed model intentionally uses six features. The shared schema also
lists `test_coverage_delta` and `author_recent_failure_rate`, but those inputs
are unavailable in the current data and were not added to the model.

## AWS Resources (M2-owned)

| Resource | Name | Status |
|----------|------|--------|
| Lambda ETL | `m2-build-feature-etl-dev` | Deployed; S3-triggered tests passed |
| DynamoDB | `m2-build-features-table` | Deployed; records verified |
| SageMaker Endpoint | `m2-xgboost-endpoint-v2-dev` | `InService` |
| Athena table | `cicdfp_m2.m2_training_features` | Deployed |
| Glue ETL | Not created | HISTORICAL/UNUSED; blocked by `AccessDeniedException` and replaced by Lambda |

## Known Limitations

1. **test_coverage_delta** is a placeholder (0.0) — raw data insufficient.
2. **time_of_day_bucket** hour boundaries are assumed, not specified in specs.md.
3. **IN_PROGRESS** treatment in failure rates is assumed (excluded from denominator).
4. **Glue was not created** because the AWS account denied access to `gluestudio-service.ap-south-1.amazonaws.com`; Lambda is the implemented ETL replacement.
5. **Author failure rate scope** (cross-repo) is assumed — not specified in specs.md.

The files `src/glue_build_feature_etl.py` and `config/glue_job_arguments.json`
are retained as historical/unused Glue attempts. They are not part of the
deployed Lambda path and must not be treated as current deployment instructions.

## Cost and Cleanup

The SageMaker endpoint incurs ongoing charges while it is `InService`; the exact
amount depends on its deployed instance type and current regional pricing.
After the demo, an approved operator can remove the endpoint without deleting
the model data:

```powershell
aws sagemaker delete-endpoint --endpoint-name m2-xgboost-endpoint-v2-dev --region ap-south-1
```

Do not run this command until the team approves cleanup. Preserve the model,
endpoint configuration, S3 data, DynamoDB table, and Athena resources unless
deletion is explicitly approved.

## Directory Structure

```
ml/
├── README.md               ← this file
├── requirements.txt        ← Python dependencies
├── conftest.py             ← pytest path setup
├── config/
│   ├── __init__.py
│   └── settings.py         ← all config from specs.md
├── data/
│   ├── synthetic/
│   │   └── raw_builds.json ← 30 synthetic M1-format events
│   └── processed/          ← output directory
├── src/
│   ├── __init__.py
│   ├── data_loader.py      ← load + validate raw events
│   └── feature_engineering.py ← raw → processed features
├── tests/
│   ├── __init__.py
│   └── test_feature_engineering.py ← 60+ test cases
├── notebooks/
│   └── m2_eda_pipeline.py  ← EDA demo script
└── models/                 ← model artifacts (local)
```
