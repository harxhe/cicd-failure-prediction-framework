import io
import json
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from src.aws_prediction import AWSIntegrationError, M2AWSService, classify_risk


FEATURES = {
    "files_changed": 177,
    "lines_changed": 117,
    "repo_recent_failure_rate": 0.0,
    "time_of_day_bucket": 1,
    "day_of_week": 5,
    "prior_failures_last_5_builds": 0,
}


def make_service(table=None, runtime=None):
    return M2AWSService(
        dynamodb_table=table or Mock(),
        sagemaker_runtime=runtime or Mock(),
    )


def test_stores_with_composite_dynamodb_key():
    table = Mock()
    service = make_service(table=table)

    item = service.store_build_features(
        repo="sample-repo", build_id="build-1", features=FEATURES
    )

    assert item["repo"] == "sample-repo"
    assert item["build_id"] == "build-1"
    table.put_item.assert_called_once_with(Item=item)


def test_retrieves_recent_records_for_repo():
    table = Mock()
    table.query.return_value = {"Items": [{"repo": "sample-repo"}]}
    service = make_service(table=table)

    result = service.get_recent_build_features(repo="sample-repo", limit=5)

    assert result == [{"repo": "sample-repo"}]
    table.query.assert_called_once_with(
        KeyConditionExpression="#repo = :repo",
        ExpressionAttributeNames={"#repo": "repo"},
        ExpressionAttributeValues={":repo": "sample-repo"},
        ScanIndexForward=False,
        Limit=5,
    )


def test_invokes_sagemaker_with_six_features_in_order():
    runtime = Mock()
    runtime.invoke_endpoint.return_value = {"Body": io.BytesIO(b"0.3551046848297119")}
    service = make_service(runtime=runtime)

    result = service.predict(build_id="build-1", features=FEATURES)

    assert result.as_dict() == {
        "build_id": "build-1",
        "risk_score": 0.3551046848297119,
        "risk_level": "low",
    }
    runtime.invoke_endpoint.assert_called_once_with(
        EndpointName="m2-xgboost-endpoint-v2-dev",
        ContentType="text/csv",
        Body=b"177,117,0,1,5,0",
    )


def test_parses_json_sagemaker_response():
    runtime = Mock()
    runtime.invoke_endpoint.return_value = {
        "Body": io.BytesIO(json.dumps({"prediction": [0.7]}).encode())
    }
    result = make_service(runtime=runtime).predict(
        build_id="build-1", features=FEATURES
    )
    assert result.risk_score == 0.7
    assert result.risk_level == "medium"


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.3999, "low"), (0.4, "medium"), (0.7, "medium"), (0.7001, "high")],
)
def test_classifies_risk(score, expected):
    assert classify_risk(score) == expected


def test_rejects_missing_feature():
    missing = dict(FEATURES)
    del missing["day_of_week"]
    with pytest.raises(ValueError, match="day_of_week"):
        make_service().predict(build_id="build-1", features=missing)


def test_rejects_non_numeric_feature():
    invalid = dict(FEATURES, lines_changed="unknown")
    with pytest.raises(ValueError, match="lines_changed"):
        make_service().predict(build_id="build-1", features=invalid)


def test_wraps_dynamodb_errors():
    table = Mock()
    table.put_item.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "PutItem",
    )
    with pytest.raises(AWSIntegrationError, match="store build features"):
        make_service(table=table).store_build_features(
            repo="sample-repo", build_id="build-1", features=FEATURES
        )


def test_wraps_sagemaker_errors():
    runtime = Mock()
    runtime.invoke_endpoint.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "invalid"}},
        "InvokeEndpoint",
    )
    with pytest.raises(AWSIntegrationError, match="invoke SageMaker"):
        make_service(runtime=runtime).predict(build_id="build-1", features=FEATURES)
