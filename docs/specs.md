# specs.md — AI-Based CI/CD Failure Prediction Framework (as built)

This document describes the system **as it was actually built and verified**, not the original plan.
Where something was not verified, it says so. Placeholders in `<angle brackets>` must be filled in
from your own AWS account (account ID, instance type, endpoint configuration name).

---

## 1. Purpose and scope

The system watches a monitored GitHub repository. Every push to `main` runs a CI build. While that
build is still running, the framework scores how likely it is to fail and sends an email alert when
the score is high.

In scope: one monitored sample repo, one prediction model, one alert channel.
Out of scope: multi-repo support, automated retraining, auto-blocking of builds, API authentication.

**What the model actually measures.** It scores the *recent failure history and change size* of the
repository, not the code itself. In practice the recent failure streak dominates the score
(section 10). Do not describe it as detecting bugs.

---

## 2. Architecture

```
 git push ─► GitHub ─► CodePipeline ─► CodeBuild (pytest)
                                           │ state change: IN_PROGRESS / SUCCEEDED / FAILED
                                           ▼
                                   EventBridge rule
                                           ▼
                        Lambda: build-event parser ──► S3  raw/{repo}/{build_id}.json
                                │                            │ S3 object-created event
                                │                            ▼
                                │                  Lambda: feature ETL ──► DynamoDB feature table
                                │ (IN_PROGRESS only)                              ▲
                                ▼                                                 │
                     Step Functions workflow                                      │
                        1. FetchFeatures (Lambda) ────────────────────────────────┘
                        2. PreparePrediction  (build CSV row)
                        3. PredictFailure  ──► SageMaker endpoint (XGBoost)
                        4. ClassifyRisk    (low / medium / high)
                        5. CheckRisk ── high ──► SNS publish ──► email
                                     └─ else ──► NoAlertNeeded (end)
```

Two things happen to every build:

1. **At build start (`IN_PROGRESS`)** the parser writes the raw event and starts the workflow. The
   score is computed from the history of *previous* builds plus the commit's size.
2. **At build end (`SUCCEEDED` / `FAILED`)** the parser overwrites the same S3 object with the final
   status. The ETL recomputes the row, and the final status becomes the row's `label`.

---

## 3. Environment and naming

| Item | Value |
|---|---|
| AWS region | `ap-south-1` (Mumbai) |
| AWS account | `<ACCOUNT_ID>` |
| Monitored repo | `harxhe/cicd-failure-prediction-sample` (branch `main`) |
| Framework repo | `harxhe/cicd-failure-prediction-framework` |
| Prefixes in use | `cicdfp-` (ingestion), `m2-` (ML), `m3-` (orchestration). Kept as built. |

| Resource | Name |
|---|---|
| CodePipeline | `cicdfp-sample-app-pipeline` (V2, execution mode SUPERSEDED, trigger: Push on `main`) |
| CodeBuild project | `cicdfp-sample-app-build` |
| EventBridge rule | `cicdfp-build-state-change-rule` |
| Parser Lambda | `cicdfp-build-event-parser-dev` |
| Data lake bucket | `cicdfp-data-lake-dev` |
| Feature ETL Lambda | `m2-build-feature-etl-dev` (S3-triggered) |
| DynamoDB table | `m2-build-features-table` (partition key `repo`, sort key `build_id`) |
| Feature lookup Lambda | `m3-fetch-features-dev` |
| State machine | `m3-cicd-workflow-dev` (JSONata) |
| SageMaker endpoint | `m2-xgboost-endpoint-v2-dev` (instance: `<INSTANCE_TYPE>`; configuration: `<ENDPOINT_CONFIG_NAME>`) |
| SNS topic | `m3-alert-topic` |
| CloudWatch | alarm `m3-build-failure-alarm-dev`, dashboard `m3-cicd-dashboard-dev` |
| Athena | database `cicdfp_m2`, table `m2_training_features` |

---

## 4. Contracts

### 4.1 Raw build event (parser Lambda writes, ETL reads)

Object key: `raw/{repo}/{build_id}.json`, where `repo` is the **CodeBuild project name**.
Each build writes the key twice: first `IN_PROGRESS`, later the final status.

```json
{
  "build_id": "uuid",
  "repo": "cicdfp-sample-app-build",
  "branch": "main",
  "commit_sha": "40-character git sha",
  "author": "git author name",
  "timestamp": "2026-10-18T06:12:51+00:00",
  "files_changed": 3,
  "lines_added": 40,
  "lines_deleted": 2,
  "test_count": 0,
  "build_duration_seconds": 27.0,
  "status": "IN_PROGRESS | SUCCEEDED | FAILED",
  "failed_stage": "BUILD | null",
  "sha_source": "codebuild | codepipeline | null",
  "enrichment_status": "ok | failed: <reason>"
}
```

Notes:
- `timestamp` is the **build start time in UTC** (from the CodeBuild event), not the time the Lambda ran.
- `commit_sha` is **not** in the CodeBuild event for pipeline-triggered builds. The parser looks it
  up through `codebuild:BatchGetBuilds` (`resolvedSourceVersion`), falling back to the CodePipeline
  source revision. `sha_source` records which one worked.
- `author`, `files_changed`, `lines_added`, `lines_deleted` come from the GitHub commits API.
- If enrichment fails, the numeric fields are written as **`0`, never `null`**, because the ETL
  calls `int()` on them. `enrichment_status` records the reason.
- `branch` is hard-coded to `main`; `test_count` is not populated (always 0).

### 4.2 Feature row (ETL writes to DynamoDB)

| Attribute | Type | Meaning |
|---|---|---|
| `repo` (key) | String | CodeBuild project name |
| `build_id` (key) | String | CodeBuild build UUID |
| `timestamp` | String | build start, ISO-8601, UTC |
| `status` | String | `IN_PROGRESS`, `SUCCEEDED`, `FAILED` |
| `label` | Number or null | `1` failed, `0` succeeded, null while running |
| `files_changed` | Number | files changed in the commit |
| `lines_changed` | Number | `lines_added + lines_deleted` |
| `repo_recent_failure_rate` | Number | failures / builds, over the up-to-5 most recent finished builds |
| `prior_failures_last_5_builds` | Number | failures among those builds (0–5) |
| `time_of_day_bucket` | Number | UTC hour: 0 = 00–05, 1 = 06–11, 2 = 12–17, 3 = 18–23 |
| `day_of_week` | Number | 0 = Monday … 6 = Sunday (UTC) |

**History rule.** For a build starting at time T, the "previous builds" are rows of the same `repo`
that have a status of `SUCCEEDED` or `FAILED`, a timestamp **strictly earlier than T** and within
the **30 days** before T. Take the **5 most recent by timestamp**.

### 4.3 Model contract (SageMaker endpoint)

- Request: `text/csv`, **no header, one row, six features in this exact order**:

  `files_changed, lines_changed, repo_recent_failure_rate, time_of_day_bucket, day_of_week, prior_failures_last_5_builds`

- Response: a single float, the failure probability (0.0–1.0).
- Training used the same order and integer encodings
  (`ml/data/sagemaker_xgboost/preprocessing_manifest.json`).

### 4.4 Risk levels (set in the workflow's `ClassifyRisk` state)

| Score | Level | Action |
|---|---|---|
| < 0.4 | low | none |
| 0.4 – 0.7 | medium | none |
| > 0.7 | high | SNS alert |

### 4.5 Workflow input and alert message

Input (sent by the parser Lambda): `{"build_id": "<uuid>", "repo": "<project name>"}`.
The feature lookup Lambda only needs `build_id`.

Alert (SNS message body, a JSON string):

```json
{"build_id": "...", "repo": "...", "risk_score": 0.86, "risk_level": "high",
 "message": "CI/CD build risk prediction completed"}
```

---

## 5. Step Functions workflow

Definition: `orchestration/workflow.asl.json`. The pre-change version is kept as
`orchestration/workflow.original.json`.

| State | Type | Behavior |
|---|---|---|
| FetchFeatures | Task (Lambda) | Looks the build up in DynamoDB. **Retry** on all errors: 5 attempts, 3 s interval, backoff 1.5, because the ETL writes the row a second or two after the workflow starts. |
| PreparePrediction | Pass | Joins the six features into one CSV string in the order of section 4.3. |
| PredictFailure | Task (SageMaker runtime `invokeEndpoint`) | Sends the CSV row, receives the score. |
| ClassifyRisk | Pass | Applies the thresholds of section 4.4. |
| CheckRisk | Choice | `risk_level = 'high'` goes to SNS, anything else to NoAlertNeeded. |
| SNS Publish | Task | Publishes the alert to `m3-alert-topic`. |
| NoAlertNeeded | Succeed | Ends; the execution output keeps the score. |

The execution **name is the build ID**, so a duplicate EventBridge delivery cannot start a second run.
The parser starts the workflow **only on the `IN_PROGRESS` event**.

---

## 6. IAM and security (as built)

Parser Lambda role (inline policies):
- `cicdfp-lambda-s3-write-raw-policy`: `s3:PutObject` on `raw/*` of the data lake bucket
- `cicdfp-lambda-build-lookup-policy`: `codebuild:BatchGetBuilds`, `codepipeline:GetPipelineState`
- `cicdfp-lambda-start-workflow-policy`: `states:StartExecution` on the state machine

Pipeline service role: extended with `codeconnections:UseConnection` for the GitHub connection
(`cicdfp-pipeline-new-connection-policy`).

Parser Lambda environment variables: `BUCKET_NAME`, `GITHUB_REPO`, `GITHUB_TOKEN`, `PIPELINE_NAME`,
`STATE_MACHINE_ARN`. Timeout: 20 seconds.

Known gaps (state these honestly in the report):
- `GITHUB_TOKEN` is a plain Lambda environment variable, not in Secrets Manager. Rotate or delete it
  after the demo.
- The API Gateway endpoint has no authentication, and is a mock (section 11).
- A KMS key was created, but which resources use it for encryption was **not verified**.
- Two SNS email subscriptions exist (yours and a teammate's). Alert bodies contain build IDs.

---

## 7. Build outcome lever (sample repo)

So that demo builds are repeatable, the sample repo's tests read `fail_flag.txt`:

| Value | Effect |
|---|---|
| `STABLE` | all tests pass (the flaky test is switched off) |
| `FAIL` | `tests/test_lever.py` fails on purpose |
| `FLAKY` | the flaky test is on: about a 30% chance the build fails |

`tools/generate_builds.py` writes this file, adds generated files (to vary `files_changed` and
`lines_changed`), commits and pushes. It refuses to run outside the sample repo.

---

## 8. Model

- Algorithm: XGBoost on SageMaker.
- Training data: TravisTorrent-derived (`ml/src/prepare_travistorrent.py`), chronological split:
  train 2,689,257 / validation 576,283 / test 576,257 rows.
- Test-split metrics (recorded by the M2 work, not re-run): accuracy 0.8225, precision 0.7157,
  **recall 0.4941**, F1 0.5846, ROC-AUC 0.8274.
- Only six features were available in the training data. `test_coverage_delta` and
  `author_recent_failure_rate` from the original design were dropped.
- The model was **not trained on this project's own builds**.

---

## 9. Data generation

17 scripted builds plus a failure streak were pushed with `tools/generate_builds.py history`
(mixed `STABLE` / `FLAKY` / `FAIL`, 1–5 generated files, 5–250 lines). The last five builds of the
script are all `FAIL`, so the next build after them is predicted from a full failure streak.

`tools/build_scores.py` (run in CloudShell) prints every build with its features, outcome and risk
score in one table, joined from DynamoDB and Step Functions.

---

## 10. Observed model behavior

Hand-made probes against the live endpoint (feature order as in section 4.3):

| Input row | Score |
|---|---|
| `0,0,0,1,5,0` (no history, empty commit) | 0.108 |
| `10,120,0,2,3,0` (120-line commit, no failures) | 0.222 |
| `0,0,0.5,1,5,2` (2 recent failures) | 0.350 |
| `20,800,0.4,2,2,2` (800 lines, 2 failures) | 0.367 |
| `3,42,0.8,1,6,4` (4 failures) | 0.507 |
| `40,3000,0.8,3,4,4` (3,000 lines, 4 failures) | 0.531 |
| `3,42,1.0,1,6,5` (5 failures) | 0.862 |
| `5,50,1.0,0,6,5` (5 failures, small commit) | 0.855 |
| `60,6000,1.0,3,4,5` (6,000 lines, 5 failures) | 0.884 |

Reading: the score climbs with the failure streak (about 0.1 with none, 0.35 with two, 0.51 with
four, 0.86 with five). Commit size is a weaker signal. A live push after five straight failed
builds scored about 0.86 and triggered the email alert. With the alert cutoff at 0.7, **an alert
needs five of the last five builds failed.**

---

## 11. Deviations from the original plan

| Original plan | As built | Reason |
|---|---|---|
| Kinesis for log streaming | Not used | EventBridge → Lambda was enough at this scale |
| Glue ETL job | Lambda ETL (`m2-build-feature-etl-dev`) | Glue was blocked by account permissions; the Glue sources remain in `ml/` as unused history |
| QuickSight dashboard | Not built | Cut for time; CloudWatch dashboard instead |
| 8 features | 6 features | Two were not available in the training data |
| EventBridge starts the workflow | Parser Lambda starts it, at build start only | Needs the build ID in plain form; avoids alerts on completed builds |
| Features as strings (`"morning"`, `"Mon"`) | Integers (see 4.2) | Matches the training encoding |
| Region `us-east-1` | `ap-south-1` | Where the account's resources were created |
| API Gateway `/predict` backed by the model | Mock only | Not completed; listed as a limitation |

---

## 12. Known issues and lessons learned

1. **Feature-lookup bug (fixed).** The deployed ETL fetched previous builds with a DynamoDB `Query`
   using `ScanIndexForward=False, Limit=5`. The sort key is a random UUID, so this returned the five
   highest UUIDs, not the five most recent builds, and recent failures were undercounted (a build
   after four straight failures stored `prior = 2`). It was found by replaying the real build history
   through both algorithms: the repo version matched 4 of 19 stored rows, the `Limit=5` version
   matched 19 of 19. Fixed by deploying the repo version (reads all rows for the repo, filters, sorts by
   timestamp). **Rows written before the fix keep their wrong feature values**; the model was not
   retrained on them.
2. **CodeBuild events carry no commit SHA** for pipeline-triggered builds (it arrived as `unknown`).
   Resolved through the CodeBuild and CodePipeline APIs (section 4.1).
3. **A null in a numeric field crashes the ETL** (`int(None)`); enrichment failures write `0`.
4. **The pipeline's push trigger and the GitHub connection** needed a new connection plus an IAM
   permission for the pipeline role before pushes started runs automatically.
5. **Recall is 0.49**: about half of failing builds are missed. The alert cutoff of 0.7 makes this
   worse for short histories.
6. The `repo` field is the CodeBuild project name, not the GitHub repository name.
7. Test data: one hand-made row lives under `repo = test-repo` (used to test the high-risk path
   without a failure streak).

---

## 13. Operations

**Cost control.** The SageMaker endpoint bills hourly while it exists and cannot be paused. Delete it
when not in use; the model and endpoint configuration stay, so it can be recreated.

```bash
aws sagemaker delete-endpoint --endpoint-name m2-xgboost-endpoint-v2-dev --region ap-south-1

aws sagemaker create-endpoint --endpoint-name m2-xgboost-endpoint-v2-dev \
  --endpoint-config-name <ENDPOINT_CONFIG_NAME> --region ap-south-1
```

While the endpoint does not exist, do not push to the sample repo (the workflow would fail at the
prediction step), or remove `STATE_MACHINE_ARN` from the parser Lambda's environment.

**Demo sequence.** Recreate the endpoint, wait for `InService`, push five failing builds
(`generate_builds.py one --mode FAIL --files 2 --lines 50`, 90 seconds apart), wait for the last to
finish, then push one build with `--mode STABLE`. Expect a score around 0.86 and an email.

**Teardown.** Delete the endpoint, rotate or delete the GitHub token, remove SNS subscriptions you no
longer need, and confirm no notebook instances are running.

---

## 14. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10 | Parser now enriches commits, uses build start time, starts the workflow | Real features and automatic prediction |
| 2026-10 | Workflow: added retry on lookup and a high-risk Choice state | Race with the ETL; avoid one email per build |
| 2026-10 | Deployed ETL replaced with the repo version | `Limit=5` by UUID bug (section 12) |
