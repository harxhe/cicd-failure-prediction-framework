import json
import boto3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from boto3.dynamodb.conditions import Key
from urllib.parse import unquote_plus

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

TABLE_NAME = "m2-build-features-table"
table = dynamodb.Table(TABLE_NAME)


def get_time_bucket(hour):
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3


def parse_timestamp(value):
    normalized = str(value).replace("Z", "+00:00")
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def get_previous_builds(repo, timestamp):
    items = []
    query_kwargs = {"KeyConditionExpression": Key("repo").eq(repo)}
    while True:
        result = table.query(**query_kwargs)
        items.extend(result.get("Items", []))
        last_key = result.get("LastEvaluatedKey")
        if not last_key:
            break
        query_kwargs["ExclusiveStartKey"] = last_key

    current_timestamp = parse_timestamp(timestamp)
    window_start = current_timestamp - timedelta(days=30)
    previous_builds = []
    for item in items:
        item_timestamp = item.get("timestamp")
        if not item_timestamp:
            continue
        try:
            parsed_timestamp = parse_timestamp(item_timestamp)
        except (TypeError, ValueError):
            continue
        if window_start <= parsed_timestamp < current_timestamp and item.get("status") in {
            "SUCCEEDED",
            "FAILED",
        }:
            previous_builds.append((parsed_timestamp, item))

    previous_builds.sort(key=lambda entry: (entry[0], entry[1].get("build_id", "")), reverse=True)
    return [item for _, item in previous_builds[:5]]


def process_record(record):
    bucket = record["s3"]["bucket"]["name"]
    key = unquote_plus(record["s3"]["object"]["key"])

    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        raw = json.loads(response["Body"].read())
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"S3 object is not valid JSON: {key}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"S3 object must contain one build object: {key}")

    required_fields = ("repo", "build_id", "timestamp", "status")
    missing_fields = [field for field in required_fields if not raw.get(field)]
    if missing_fields:
        raise ValueError(
            f"Build event is missing required fields: {', '.join(missing_fields)}"
        )

    repo = raw["repo"]
    build_id = raw["build_id"]
    timestamp = raw["timestamp"]
    dt = parse_timestamp(timestamp)

    files_changed = int(raw.get("files_changed", 0))
    lines_changed = int(raw.get("lines_added", 0)) + int(raw.get("lines_deleted", 0))
    previous_builds = get_previous_builds(repo, timestamp)
    failures = sum(1 for item in previous_builds if item.get("status") == "FAILED")
    repo_recent_failure_rate = Decimal(str(failures / len(previous_builds))) if previous_builds else Decimal("0")
    status = raw.get("status")
    label = 1 if status == "FAILED" else 0 if status == "SUCCEEDED" else None

    item = {
        "repo": repo,
        "build_id": build_id,
        "timestamp": timestamp,
        "files_changed": files_changed,
        "lines_changed": lines_changed,
        "repo_recent_failure_rate": repo_recent_failure_rate,
        "time_of_day_bucket": get_time_bucket(dt.hour),
        "day_of_week": dt.weekday(),
        "prior_failures_last_5_builds": failures,
        "label": label,
        "status": status or "UNKNOWN",
    }
    table.put_item(Item=item)
    return item


def lambda_handler(event, context):
    processed = [process_record(record) for record in event["Records"]]

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Feature ETL completed",
            "build_ids": [item["build_id"] for item in processed],
            "processed_count": len(processed),
        })
    }