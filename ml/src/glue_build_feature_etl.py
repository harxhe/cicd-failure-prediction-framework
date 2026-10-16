"""HISTORICAL/UNUSED Glue ETL attempt for real M1 CodeBuild records.

Glue creation was denied by AWS. The deployed ETL replacement is
``src/lambda/m2_build_feature_etl.py``; this module is retained for history and
must not be treated as the current deployment path.

Reads JSON objects from the real raw S3 prefix and writes JSON Lines features
to a dedicated processed prefix. Historical features use only records with a
strictly earlier timestamp than the current build.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import boto3
from awsglue.utils import getResolvedOptions


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def parse_timestamp(value):
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def label_for_status(status):
    if status == "FAILED":
        return 1
    if status == "SUCCEEDED":
        return 0
    return None


def time_bucket(timestamp):
    if timestamp is None:
        return None
    if timestamp.hour < 6:
        return "night"
    if timestamp.hour < 12:
        return "morning"
    if timestamp.hour < 18:
        return "afternoon"
    return "evening"


def number(record, field):
    value = record.get(field)
    return float(value) if value is not None else None


def load_records(s3_client, source_uri):
    bucket, prefix = parse_s3_uri(source_uri)
    records = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
            payload = json.loads(body.decode("utf-8"))
            if isinstance(payload, list):
                records.extend(payload)
            else:
                records.append(payload)
    return records


def engineer(records):
    prepared = []
    for record in records:
        timestamp = parse_timestamp(record.get("timestamp"))
        if not record.get("build_id") or not record.get("repo") or timestamp is None:
            LOGGER.warning("Skipping record missing build_id, repo, or timestamp")
            continue
        prepared.append({
            "raw": record,
            "timestamp": timestamp,
            "label": label_for_status(record.get("status")),
        })

    prepared.sort(key=lambda item: (item["timestamp"], item["raw"]["build_id"]))
    features = []
    for index, current in enumerate(prepared):
        raw = current["raw"]
        current_timestamp = current["timestamp"]
        history = [
            item for item in prepared[:index]
            if item["timestamp"] < current_timestamp
            and item["raw"].get("status") in {"SUCCEEDED", "FAILED"}
        ]
        repo_history = [
            item for item in history if item["raw"].get("repo") == raw.get("repo")
        ]
        window_start = current_timestamp - timedelta(days=30)
        recent = [item for item in repo_history if item["timestamp"] >= window_start]
        previous_five = repo_history[-5:]
        known_author = raw.get("author") not in {None, "", "unknown", "UNKNOWN"}
        author_history = [
            item for item in history if item["raw"].get("author") == raw.get("author")
        ] if known_author else []

        lines_added = number(raw, "lines_added")
        lines_deleted = number(raw, "lines_deleted")
        features.append({
            "build_id": raw["build_id"],
            "repo": raw["repo"],
            "files_changed": number(raw, "files_changed"),
            "lines_changed": (
                lines_added + lines_deleted
                if lines_added is not None and lines_deleted is not None else None
            ),
            "test_coverage_delta": None,
            "author_recent_failure_rate": (
                sum(item["label"] == 1 for item in author_history) / len(author_history)
                if author_history else None
            ),
            "repo_recent_failure_rate": (
                sum(item["label"] == 1 for item in recent) / len(recent)
                if recent else 0.0
            ),
            "time_of_day_bucket": time_bucket(current_timestamp),
            "day_of_week": current_timestamp.strftime("%a"),
            "prior_failures_last_5_builds": sum(item["label"] == 1 for item in previous_five),
            "label": current["label"],
        })
    return features


def main():
    args = getResolvedOptions(
        __import__("sys").argv,
        ["SOURCE_PATH", "OUTPUT_PATH"],
    )
    s3 = boto3.client("s3")
    output_bucket, output_prefix = parse_s3_uri(args["OUTPUT_PATH"])
    records = load_records(s3, args["SOURCE_PATH"])
    features = engineer(records)
    body = "".join(json.dumps(feature, separators=(",", ":")) + "\n" for feature in features)
    output_key = output_prefix.rstrip("/") + "/part-00000.jsonl"
    s3.put_object(
        Bucket=output_bucket,
        Key=output_key,
        Body=body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    LOGGER.info("Processed %d raw records into s3://%s/%s", len(records), output_bucket, output_key)


if __name__ == "__main__":
    main()