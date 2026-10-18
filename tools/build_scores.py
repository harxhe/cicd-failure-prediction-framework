"""
build_scores.py: one table with every build's features, outcome, and risk score.

Run it in AWS CloudShell (ap-south-1), which already has your credentials:
  Actions -> Upload file -> build_scores.py, then:  python3 build_scores.py

It joins two sources by build id:
  - DynamoDB m2-build-features-table  (features and final build status)
  - Step Functions executions         (risk_score and risk_level)
"""
import json
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

REGION = "ap-south-1"
STATE_MACHINE_ARN = "arn:aws:states:ap-south-1:880690594500:stateMachine:m3-cicd-workflow-dev"
TABLE = "m2-build-features-table"
REPO = "cicdfp-sample-app-build"

sf = boto3.client("stepfunctions", region_name=REGION)
table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)


def classify_output(execution_arn):
    """Read the ClassifyRisk state's output from the execution history."""
    events = sf.get_execution_history(
        executionArn=execution_arn, reverseOrder=True, maxResults=100
    )["events"]
    for event in events:
        details = event.get("stateExitedEventDetails")
        if event["type"] == "PassStateExited" and details and details.get("name") == "ClassifyRisk":
            return json.loads(details["output"])
    return {}


def load_scores():
    scores = {}
    paginator = sf.get_paginator("list_executions")
    for page in paginator.paginate(stateMachineArn=STATE_MACHINE_ARN):
        for ex in page["executions"]:
            out = {}
            if ex["status"] == "SUCCEEDED":
                detail = sf.describe_execution(executionArn=ex["executionArn"])
                try:
                    out = json.loads(detail.get("output") or "{}")
                except ValueError:
                    out = {}
                if "risk_score" not in out:  # high-risk runs end with the SNS response
                    out = classify_output(ex["executionArn"])
            scores[ex["name"]] = (ex["status"], out.get("risk_score"), out.get("risk_level"))
    return scores


def load_builds():
    items, kwargs = [], {"KeyConditionExpression": Key("repo").eq(REPO)}
    while True:
        result = table.query(**kwargs)
        items.extend(result.get("Items", []))
        if not result.get("LastEvaluatedKey"):
            break
        kwargs["ExclusiveStartKey"] = result["LastEvaluatedKey"]
    # Skip partial or empty rows left over from tests
    return sorted((i for i in items if i.get("timestamp") and i.get("status")),
                  key=lambda i: i["timestamp"])


def num(value, digits=2):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".") if digits else str(int(value))


def main():
    scores = load_scores()
    builds = load_builds()

    print(f"{'time (UTC)':<17} {'build':<9} {'result':<10} {'files':>5} {'lines':>5} "
          f"{'rate':>5} {'prior':>5}  {'score':>6} {'level':<7} {'workflow'}")
    for b in builds:
        wf_status, score, level = scores.get(b["build_id"], (None, None, None))
        print(f"{b['timestamp'][5:16].replace('T', ' '):<17} {b['build_id'][:8]:<9} "
              f"{b['status']:<10} {num(b.get('files_changed'), 0):>5} {num(b.get('lines_changed'), 0):>5} "
              f"{num(b.get('repo_recent_failure_rate')):>5} {num(b.get('prior_failures_last_5_builds'), 0):>5}  "
              f"{('%.3f' % score) if isinstance(score, (int, float, Decimal)) else '-':>6} "
              f"{level or '-':<7} {wf_status or 'no run'}")

    done = [b for b in builds if b["status"] in ("SUCCEEDED", "FAILED")]
    last5 = done[-5:]
    failed = sum(1 for b in last5 if b["status"] == "FAILED")
    print(f"\nLast {len(last5)} finished builds (oldest first): "
          + ", ".join(b["status"][0] for b in last5))
    print(f"The next push will see prior_failures_last_5_builds={failed}, "
          f"repo_recent_failure_rate={failed / len(last5):.2f}" if last5 else "No finished builds.")


if __name__ == "__main__":
    main()