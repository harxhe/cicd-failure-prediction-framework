import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "lambda" / "m2_build_feature_etl.py"
SPEC = importlib.util.spec_from_file_location("m2_build_feature_etl", MODULE_PATH)
lambda_etl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(lambda_etl)


def s3_record(key):
    return {
        "s3": {
            "bucket": {"name": "cicdfp-data-lake-dev"},
            "object": {"key": key},
        }
    }


def build_event(build_id, timestamp, status):
    return {
        "build_id": build_id,
        "repo": "sample-repo",
        "timestamp": timestamp,
        "files_changed": 4,
        "lines_added": 10,
        "lines_deleted": 2,
        "status": status,
    }


def test_lambda_processes_all_records_and_preserves_in_progress_label(monkeypatch):
    table = Mock()
    table.query.return_value = {
        "Items": [
            {"build_id": "future", "timestamp": "2026-10-10T13:00:00Z", "status": "FAILED", "label": 1},
            {"build_id": "old-success", "timestamp": "2026-10-10T09:00:00Z", "status": "SUCCEEDED", "label": 0},
            {"build_id": "old-failure", "timestamp": "2026-10-10T10:00:00Z", "status": "FAILED", "label": 1},
            {"build_id": "oldest-success", "timestamp": "2026-10-10T08:00:00Z", "status": "SUCCEEDED", "label": 0},
            {"build_id": "old-success-2", "timestamp": "2026-10-10T11:00:00Z", "status": "SUCCEEDED", "label": 0},
        ]
    }
    s3 = Mock()
    s3.get_object.side_effect = [
        {"Body": io.BytesIO(json.dumps(build_event("build-1", "2026-10-10T12:00:00Z", "FAILED")).encode())},
        {"Body": io.BytesIO(json.dumps(build_event("build-2", "2026-10-10T14:00:00Z", "IN_PROGRESS")).encode())},
    ]
    monkeypatch.setattr(lambda_etl, "table", table)
    monkeypatch.setattr(lambda_etl, "s3", s3)

    result = lambda_etl.lambda_handler(
        {"Records": [s3_record("raw/build-1.json"), s3_record("raw/build-2.json")]},
        None,
    )

    first_item, second_item = [call.kwargs["Item"] for call in table.put_item.call_args_list]
    assert result["statusCode"] == 200
    assert json.loads(result["body"]) == {
        "message": "Feature ETL completed",
        "build_ids": ["build-1", "build-2"],
        "processed_count": 2,
    }
    assert first_item["prior_failures_last_5_builds"] == 1
    assert float(first_item["repo_recent_failure_rate"]) == 1 / 4
    assert first_item["label"] == 1
    assert second_item["label"] is None
    assert s3.get_object.call_count == 2
    assert table.put_item.call_count == 2


def test_get_previous_builds_reads_all_pages_and_sorts_by_timestamp(monkeypatch):
    table = Mock()
    table.query.side_effect = [
        {
            "Items": [
                {"build_id": "z", "timestamp": "2026-10-09T10:00:00Z", "status": "FAILED"},
            ],
            "LastEvaluatedKey": {"repo": "sample-repo", "build_id": "z"},
        },
        {
            "Items": [
                {"build_id": "a", "timestamp": "2026-10-09T11:00:00Z", "status": "SUCCEEDED"},
                {"build_id": "future", "timestamp": "2026-10-11T10:00:00Z", "status": "FAILED"},
                {"build_id": "old", "timestamp": "2026-09-01T10:00:00Z", "status": "FAILED"},
            ]
        },
    ]
    monkeypatch.setattr(lambda_etl, "table", table)

    previous = lambda_etl.get_previous_builds("sample-repo", "2026-10-10T10:00:00Z")

    assert [item["build_id"] for item in previous] == ["a", "z"]
    assert table.query.call_count == 2
    assert table.query.call_args_list[1].kwargs["ExclusiveStartKey"] == {
        "repo": "sample-repo",
        "build_id": "z",
    }


def test_duplicate_event_overwrites_same_dynamodb_key(monkeypatch):
    table = Mock()
    table.query.return_value = {"Items": []}
    s3 = Mock()
    payload = build_event("duplicate-build", "2026-10-10T12:00:00Z", "SUCCEEDED")
    s3.get_object.side_effect = [
        {"Body": io.BytesIO(json.dumps(payload).encode())},
        {"Body": io.BytesIO(json.dumps(payload).encode())},
    ]
    monkeypatch.setattr(lambda_etl, "table", table)
    monkeypatch.setattr(lambda_etl, "s3", s3)

    lambda_etl.lambda_handler(
        {"Records": [s3_record("raw/duplicate.json"), s3_record("raw/duplicate.json")]},
        None,
    )

    assert table.put_item.call_count == 2
    assert table.put_item.call_args_list[0].kwargs["Item"] == table.put_item.call_args_list[1].kwargs["Item"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "not valid JSON"),
        ({"repo": "sample-repo"}, "missing required fields"),
    ],
)
def test_rejects_malformed_or_incomplete_build_event(monkeypatch, payload, message):
    table = Mock()
    s3 = Mock()
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    s3.get_object.return_value = {"Body": io.BytesIO(body)}
    monkeypatch.setattr(lambda_etl, "table", table)
    monkeypatch.setattr(lambda_etl, "s3", s3)

    with pytest.raises(ValueError, match=message):
        lambda_etl.lambda_handler({"Records": [s3_record("raw/invalid.json")]}, None)

    table.put_item.assert_not_called()
