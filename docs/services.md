# services.md — Services used in this project

Region: `ap-south-1` (Mumbai). Resource names are the ones created in the console.

## 1. Services in the running system

| # | Service | Resource | What it does here |
|---|---|---|---|
| 1 | **GitHub** (not AWS) | `cicd-failure-prediction-sample`, `cicd-failure-prediction-framework` | Source of the monitored code; its REST API supplies commit author and diff size |
| 2 | **AWS CodeConnections** | GitHub App connection | Lets CodePipeline pull from GitHub and receive push events |
| 3 | **CodePipeline** | `cicdfp-sample-app-pipeline` | Starts a run on every push to `main`: Source, then Build |
| 4 | **CodeBuild** | `cicdfp-sample-app-build` | Runs `pytest` in a container; its pass/fail is the signal the whole system learns from |
| 5 | **EventBridge** | `cicdfp-build-state-change-rule` | Captures CodeBuild state changes (start, success, failure) and invokes the parser |
| 6 | **Lambda** | `cicdfp-build-event-parser-dev` | Turns a build event into an enriched JSON record, writes it to S3, starts the workflow at build start |
| 7 | **Lambda** | `m2-build-feature-etl-dev` | S3-triggered ETL that computes the six model features and writes a DynamoDB row |
| 8 | **Lambda** | `m3-fetch-features-dev` | Looks up a build's features in DynamoDB for the workflow |
| 9 | **S3** | `cicdfp-data-lake-dev` (plus the auto-created CodePipeline artifact bucket) | Data lake: raw build records, processed training data |
| 10 | **DynamoDB** | `m2-build-features-table` | Feature store: one row per build, read at prediction time |
| 11 | **Step Functions** | `m3-cicd-workflow-dev` | Orchestrates lookup, prediction, risk classification and the alert decision |
| 12 | **SageMaker** | XGBoost model, endpoint `m2-xgboost-endpoint-v2-dev` | Trains the model and serves real-time failure probabilities (endpoint is deleted when idle) |
| 13 | **SNS** | `m3-alert-topic` | Emails the high-risk alert |
| 14 | **CloudWatch** | log groups, alarm `m3-build-failure-alarm-dev`, dashboard `m3-cicd-dashboard-dev` | Lambda and build logs, failure alarm, monitoring dashboard |
| 15 | **Athena** (with the Glue Data Catalog) | database `cicdfp_m2`, table `m2_training_features` | SQL over the training data in S3 for exploration and checks |
| 16 | **API Gateway** | REST API with a `POST` route | **Mock only**; not connected to the model |
| 17 | **IAM** | service roles, inline policies per Lambda | Least-privilege access between services |
| 18 | **KMS** | project key | Created; which resources it encrypts was not verified |

## 2. Operational tools (not part of the data path)

| Tool | Use |
|---|---|
| **CloudShell** | Runs `build_scores.py` and endpoint checks with console credentials |
| **Resource Groups & Tag Editor** | Listed every resource in the region for an inventory audit |
| **Billing / Budgets** | Cost tracking; the SageMaker endpoint is the only significant cost |

## 3. Planned but not used

| Service | Why it was dropped |
|---|---|
| Kinesis | EventBridge to Lambda handles the event volume; streaming was unnecessary |
| Glue ETL job | Blocked by account permissions; replaced by the ETL Lambda (Glue sources remain in `ml/` as history) |
| QuickSight | Cut for time; CloudWatch dashboard instead |
| SageMaker Feature Store / Pipelines | Out of scope; DynamoDB serves as the feature store |
| Custom VPC / NAT gateway | Not needed; only the default VPC exists |

## 4. Who owns what

| Layer | Services |
|---|---|
| Data pipeline | CodeConnections, CodePipeline, CodeBuild, EventBridge, parser Lambda, S3 |
| ML and prediction | ETL Lambda, DynamoDB, Athena, SageMaker |
| Orchestration and operations | Step Functions, feature lookup Lambda, SNS, CloudWatch, API Gateway, IAM, KMS |

## 5. Cost note

Everything above costs pennies at this scale except the **SageMaker real-time endpoint**, which bills
every hour it exists. It cannot be paused, only deleted and recreated from the saved model and
endpoint configuration.
