# CI/CD Failure Prediction Framework

A serverless AWS system that predicts whether a CI build is likely to fail **while the build is still
running**, and emails an alert when the risk is high.

This repository holds the framework. The application it monitors lives in a separate repository:
[`cicd-failure-prediction-sample`](https://github.com/harxhe/cicd-failure-prediction-sample).

> Course project for Cloud Architecture. Built and verified in AWS region `ap-south-1`.

## How it works

```
git push -> GitHub -> CodePipeline -> CodeBuild (pytest)
                                          | build state change
                                          v
                                  EventBridge -> Lambda (parser) -> S3 raw/*.json
                                                      |                 | S3 event
                                                      |                 v
                                                      |        Lambda (feature ETL) -> DynamoDB
                                                      | at build start
                                                      v
                                          Step Functions workflow
                                            FetchFeatures -> SageMaker endpoint -> classify risk
                                                         high risk? -> SNS email
```

1. A push to `main` of the sample repo starts a build.
2. The parser Lambda turns each build event into an enriched JSON record (commit author, files and
   lines changed from the GitHub API) and stores it in S3.
3. A second Lambda turns each record into a feature row in DynamoDB (recent failure rate, prior
   failures, time of day, commit size).
4. At build start, a Step Functions workflow loads the features, asks an XGBoost model on SageMaker for
   a failure probability, and sends an SNS email when it is above 0.7.
5. When the build ends, the same record is updated with the real outcome, which becomes the label.

## What the model measures

It scores **recent failure history and commit size**, not the code. A failure streak dominates the
score: about 0.1 with no recent failures, 0.35 with two, 0.51 with four and 0.86 with five. With the
alert cutoff at 0.7, an alert needs five failed builds in a row. Recall on the test split is 0.49.
Full numbers and the model's limits are in [`docs/specs.md`](docs/specs.md).

## Repository layout

```
.
├── ingestion/
│   └── parser_lambda.py            # CodeBuild event -> enriched JSON in S3, starts the workflow
├── ml/                             # ML layer: data prep, XGBoost prep, ETL Lambda, tests, reports
│   ├── src/lambda/m2_build_feature_etl.py
│   ├── MODEL_EVALUATION.md
│   └── README.md
├── orchestration/
│   ├── workflow.asl.json           # Step Functions definition (current)
│   ├── workflow.original.json      # definition before the retry/choice changes
│   └── fetch_features_lambda.py    # feature lookup Lambda (copy of the console version)
├── tools/
│   ├── generate_builds.py          # pushes controlled passing/failing builds to the sample repo
│   └── build_scores.py             # run in CloudShell: one table of builds, features and scores
└── docs/
    ├── specs.md                    # as-built specification (start here)
    ├── services.md                 # every AWS service used, and why
    ├── screenshots.md              # demo screenshot checklist
    └── evidence/                   # screenshots
```

## AWS services

CodePipeline, CodeBuild, CodeConnections, EventBridge, Lambda, S3, DynamoDB, Step Functions,
SageMaker, SNS, CloudWatch, Athena (with the Glue Data Catalog), API Gateway (mock only), IAM and KMS.
See [`docs/services.md`](docs/services.md) for what each one does here.

## Setting it up

The resources were created in the AWS console, not with infrastructure-as-code. To rebuild, create
them in this order (names and contracts are in `docs/specs.md`):

1. Sample repo, CodeBuild project and CodePipeline with a GitHub connection and a push trigger on `main`.
2. S3 data lake bucket and the EventBridge rule on CodeBuild state changes.
3. Parser Lambda (`ingestion/parser_lambda.py`) with its environment variables, a 20-second timeout and
   the IAM policies listed in `docs/specs.md` section 6.
4. DynamoDB table (`repo` partition key, `build_id` sort key) and the ETL Lambda with an S3 trigger on
   `raw/`.
5. SageMaker model and endpoint (see `ml/README.md`).
6. Feature lookup Lambda, SNS topic with an email subscription, and the Step Functions state machine
   from `orchestration/workflow.asl.json`.
7. Set `STATE_MACHINE_ARN` on the parser Lambda.

## Running the demo

1. Recreate the SageMaker endpoint and wait for `InService`.
2. In the sample repo, push five failing builds, 90 seconds apart:
   ```
   python tools/generate_builds.py one --mode FAIL --files 2 --lines 50
   ```
3. Wait for the last one to finish, then push one normal build:
   ```
   python tools/generate_builds.py one --mode STABLE --files 2 --lines 40
   ```
4. Watch the Step Functions execution; expect a score around 0.86 and an email.
5. In AWS CloudShell, run `python3 tools/build_scores.py` to see every build with its score.

`generate_builds.py` must be run from inside the **sample** repo; it refuses to run anywhere else.

## Cost

The SageMaker endpoint bills hourly and cannot be paused. Delete it when you are not using it and
recreate it from the saved endpoint configuration (commands in `docs/specs.md` section 13). Everything
else in this project costs very little at this scale.

## Known limitations

- The model was trained on TravisTorrent data, not on this project's own builds.
- Recall is 0.49; about half of failing builds are not flagged.
- Two of the originally planned features are not available, so the model uses six.
- The API Gateway endpoint is a mock and has no authentication.
- The GitHub token is stored as a Lambda environment variable.
- Features of rows written before the ETL fix are understated; see `docs/specs.md` section 12.

## Team

Built by Harsh Vardhan with teammates who built the ML layer (feature pipeline, SageMaker model) and the
orchestration layer (Step Functions, SNS, CloudWatch).
