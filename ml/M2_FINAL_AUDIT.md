# M2 Final Implementation Audit

Audit date: 2026-10-10. Local code and artifacts were inspected directly. AWS
resource status below uses the supplied deployment evidence; no AWS mutation was
performed and no credentials are included.

## Requirement Review

| Requirement | Expected behavior | Actual behavior | Status | Evidence | Required action |
|---|---|---|---|---|---|
| Backup dataset and EDA | Use a backup dataset when real history is too small; document features | TravisTorrent mapping, EDA, missing-feature analysis, and report exist | Complete | `ml/DATA_MAPPING.md`, `ml/EDA_REPORT.md` | None |
| Raw schema validation | Enforce M1 identity/status fields and locked schema | Local loader validates fields, nulls, extras, and statuses; Lambda validates fields it consumes | Complete locally | `ml/src/data_loader.py`, `ml/src/lambda/m2_build_feature_etl.py`, tests | Confirm shared schema remains approved |
| Feature derivation | Derive files, lines, repo history, time bucket, weekday, prior failures, label | Lambda and local pipeline derive six deployed features; `IN_PROGRESS` is unlabeled | Complete for deployed model | Lambda source and regression tests | Team approval for six-feature contract |
| Historical leakage prevention | Use only strictly earlier completed builds | Lambda parses timestamps, excludes future/equal timestamps, paginates, sorts newest-first, and limits to five | Complete locally | `get_previous_builds()` and tests | Live AWS replay evidence remains external |
| Glue ETL (Day 3 plan) | Glue reads raw S3 and writes processed features | Glue creation failed with supplied service-side `AccessDeniedException`; Lambda replacement reads S3 and writes DynamoDB | Replaced, approved needed | `ml/src/lambda/m2_build_feature_etl.py`; supplied AWS evidence | Approve Lambda replacement and update shared spec |
| Athena table | Query processed training/features data | `cicdfp_m2.m2_training_features` is supplied as deployed | AWS reported complete; schema unverified locally | Supplied AWS state; no DDL/schema file in workspace | Provide/approve Athena DDL and source location |
| DynamoDB feature store | Store repo/build records for recent history | `m2-build-features-table` is supplied as deployed; Lambda writes items with repo/build_id keys | AWS reported complete; local contract tested | Lambda source, `test_aws_prediction.py`, supplied AWS evidence | Verify table encryption/key schema live |
| SageMaker training | Train on processed features | Local model artifact exists; six-feature SageMaker CSV preprocessing exists | Complete locally; job provenance not in workspace | `xgboost-model`, `ml/data/sagemaker_xgboost/`, manifests | Record training job/model provenance |
| SageMaker endpoint | Deploy real-time/serverless inference | `m2-xgboost-endpoint-v2-dev` supplied `InService` | AWS reported complete | Supplied AWS state; local adapter tests | Confirm live response shape with M3 |
| Prediction payload | Send deployed six features in model order | Local adapter sends `files_changed, lines_changed, repo_recent_failure_rate, time_of_day_bucket, day_of_week, prior_failures_last_5_builds` as CSV | Complete locally | `ml/src/aws_prediction.py`, tests | Keep synchronized with deployed endpoint |
| Prediction response / Choice | Return `risk_score` and lowercase `low/medium/high`; Choice matches high | Local adapter now returns lowercase labels and fixed thresholds | Complete locally; live M3 unverified | `ml/src/aws_prediction.py`, tests | M3 must verify Choice uses `high` |
| Evaluation | Report accuracy, precision, recall, F1, confusion matrix, feature importance | Reproduced on 576,257-row chronological test split; metrics match report | Complete locally | `MODEL_EVALUATION.md`, `ml/data/splits/manifest.json` | Decide whether 49.4% recall is acceptable |
| Day 6 testing | Run 8-10 pass/fail builds through full pipeline | Local tests and supplied Lambda/S3 tests exist; full M1→M3→SNS evidence is not in workspace | Partial | 97 local tests; supplied manual Lambda evidence | Record complete end-to-end alert run |
| IAM least privilege | Scoped Lambda/SageMaker permissions, no hardcoded credentials | No IAM policy, role, KMS, or CloudFormation artifacts are present locally | Unverified | `boto3` default role usage; no policy files | M3 provide live policy/encryption evidence |
| Encryption | KMS protects S3/DynamoDB at rest | Shared spec requires KMS, but no local/live evidence was supplied | Unverified | `specs.md` only | Team/M3 verify KMS configuration |
| Documentation and cost cleanup | Setup, tests, limitations, teardown/cost note | README and this audit document cover setup, architecture, limitations, and endpoint cleanup command | Complete locally | `ml/README.md`, this file | Execute cleanup only after demo approval |

## Deployed Six Features vs Shared Eight-Feature Schema

The deployed model and Lambda use these six ordered features:

```text
files_changed,
lines_changed,
repo_recent_failure_rate,
time_of_day_bucket,
day_of_week,
prior_failures_last_5_builds
```

The shared schema additionally specifies `test_coverage_delta` and
`author_recent_failure_rate`. The repository documents that coverage percentage
and stable author identity are unavailable in the available data. They were not
silently invented or added to the deployed model.

### Team decision request

Approve one of these explicit choices before changing shared contracts:

1. Accept the deployed six-feature model as the current demo contract and update
   the shared schema/architecture to identify the two unavailable features; or
2. Supply real coverage and author data, retrain, redeploy, and update all
   consumers together.

This audit does not edit `specs.md`, change the model, or alter AWS resources.

## Data Storage and Column Provenance

The Lambda source reads each S3 raw JSON object and writes the derived record to
DynamoDB. It does **not** write processed data to S3. The Athena table is reported
as deployed, but its DDL and physical source location are not present locally;
this audit therefore does not claim whether Athena reads DynamoDB, an S3 export,
or another registered source.

The local SageMaker CSV artifacts are headerless and use this exact order:

```text
label,files_changed,lines_changed,repo_recent_failure_rate,time_of_day_bucket,day_of_week,prior_failures_last_5_builds
```

The split manifest records chronological TravisTorrent provenance, 70/15/15
membership, 3,841,797 total rows, and 576,257 test rows. The saved model was
re-evaluated against `day6-test.csv`; the results match `MODEL_EVALUATION.md`.

## IAM, Encryption, and Cost Evidence

No local IAM, KMS, Athena DDL, endpoint instance-type, or deployment-template
evidence was found. The known AWS state is therefore reported as supplied, not
independently queried here. The source uses boto3's ambient execution role and
contains no credentials.

The SageMaker endpoint incurs ongoing charges while `InService`; the exact cost
requires its deployed instance type and region pricing. After the demo, an
approved operator can stop the cost by deleting only the endpoint:

```powershell
aws sagemaker delete-endpoint --endpoint-name m2-xgboost-endpoint-v2-dev --region ap-south-1
```

Do not run cleanup until the team approves it. Preserve the model, endpoint
configuration, S3 data, DynamoDB table, and Athena resources unless the team
explicitly authorizes their deletion.

## Validation Evidence

AWS CLI access is configured, but the current identity is not authorized to
read project Lambda, DynamoDB, S3, Athena, or SageMaker resource metadata. The
CLI default region is `eu-north-1`; project checks were attempted explicitly in
`ap-south-1` and returned `AccessDenied`. Live resource properties are therefore
unverified unless identified as supplied deployment evidence.

The final local command was:

```powershell
cd ml
python -m pytest tests -q
```

Result: **97 passed** after the additional Lambda and prediction regression
coverage. The original baseline was **92 passed**; the prior Lambda regression
file added two tests, and this audit added two parametrized malformed/incomplete
input cases plus one duplicate-event case. Prediction expectations were updated
for lowercase labels without changing the original test count.

Unit tests use mocks and do not prove live AWS permissions, encryption, Athena
schema, endpoint status, or the complete M1→M3→SNS workflow.
