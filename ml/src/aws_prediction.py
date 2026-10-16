"""M2 DynamoDB feature storage and SageMaker prediction service."""

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config.settings import (
    AWS_REGION,
    DYNAMODB_TABLE,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    SAGEMAKER_ENDPOINT,
)


MODEL_FEATURE_ORDER = (
    "files_changed",
    "lines_changed",
    "repo_recent_failure_rate",
    "time_of_day_bucket",
    "day_of_week",
    "prior_failures_last_5_builds",
)


class AWSIntegrationError(RuntimeError):
    """Raised when an AWS feature-storage or inference operation fails."""


@dataclass(frozen=True)
class PredictionResult:
    build_id: str
    risk_score: float
    risk_level: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
        }


class M2AWSService:
    """Owns M2's DynamoDB records and SageMaker endpoint calls."""

    def __init__(
        self,
        *,
        dynamodb_table: Any | None = None,
        sagemaker_runtime: Any | None = None,
        region_name: str = AWS_REGION,
        table_name: str = DYNAMODB_TABLE,
        endpoint_name: str = SAGEMAKER_ENDPOINT,
    ) -> None:
        self._table = dynamodb_table or boto3.resource(
            "dynamodb", region_name=region_name
        ).Table(table_name)
        self._runtime = sagemaker_runtime or boto3.client(
            "sagemaker-runtime", region_name=region_name
        )
        self._endpoint_name = endpoint_name

    def store_build_features(
        self,
        *,
        repo: str,
        build_id: str,
        features: Mapping[str, Any],
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store one feature record using the table's composite primary key."""
        if not repo or not build_id:
            raise ValueError("repo and build_id are required")

        item = {"repo": repo, "build_id": build_id, **dict(features)}
        if extra_attributes:
            item.update(extra_attributes)

        try:
            self._table.put_item(Item=item)
        except (BotoCoreError, ClientError) as exc:
            raise AWSIntegrationError("Unable to store build features") from exc
        return item

    def get_build_features(self, *, repo: str, build_id: str) -> dict[str, Any] | None:
        """Retrieve one feature record by its DynamoDB primary key."""
        try:
            response = self._table.get_item(
                Key={"repo": repo, "build_id": build_id}
            )
        except (BotoCoreError, ClientError) as exc:
            raise AWSIntegrationError("Unable to retrieve build features") from exc
        return response.get("Item")

    def get_recent_build_features(
        self, *, repo: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve up to ``limit`` records for a repository."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        try:
            response = self._table.query(
                KeyConditionExpression="#repo = :repo",
                ExpressionAttributeNames={"#repo": "repo"},
                ExpressionAttributeValues={":repo": repo},
                ScanIndexForward=False,
                Limit=limit,
            )
        except (BotoCoreError, ClientError) as exc:
            raise AWSIntegrationError("Unable to retrieve recent build features") from exc
        return response.get("Items", [])

    def predict(
        self, *, build_id: str, features: Mapping[str, Any]
    ) -> PredictionResult:
        """Invoke SageMaker with the exact six-feature model contract."""
        values = _ordered_model_values(features)
        payload = ",".join(_csv_value(value) for value in values)
        try:
            response = self._runtime.invoke_endpoint(
                EndpointName=self._endpoint_name,
                ContentType="text/csv",
                Body=payload.encode("utf-8"),
            )
            score = _parse_prediction(response["Body"])
        except (BotoCoreError, ClientError, KeyError, ValueError, TypeError) as exc:
            raise AWSIntegrationError("Unable to invoke SageMaker endpoint") from exc

        return PredictionResult(
            build_id=build_id,
            risk_score=score,
            risk_level=classify_risk(score),
        )


def classify_risk(risk_score: float) -> str:
    """Map a model score to the lowercase levels used by M3's Choice state."""
    score = _finite_number(risk_score, "risk_score")
    if score < RISK_THRESHOLD_LOW:
        return "low"
    if score <= RISK_THRESHOLD_HIGH:
        return "medium"
    return "high"


def _ordered_model_values(features: Mapping[str, Any]) -> list[Any]:
    missing = [name for name in MODEL_FEATURE_ORDER if name not in features]
    if missing:
        raise ValueError(f"Missing model features: {', '.join(missing)}")
    values = [features[name] for name in MODEL_FEATURE_ORDER]
    for name, value in zip(MODEL_FEATURE_ORDER, values):
        _finite_number(value, name)
    return values


def _csv_value(value: Any) -> str:
    _finite_number(value, "model feature")
    if isinstance(value, float):
        return str(value).rstrip("0").rstrip(".")
    return str(value)


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _parse_prediction(body: Any) -> float:
    raw = body.read() if hasattr(body, "read") else body
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(parsed, dict):
        for key in ("risk_score", "prediction", "score"):
            if key in parsed:
                parsed = parsed[key]
                break
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("SageMaker response contained no prediction")
        parsed = parsed[0]
    return _finite_number(parsed, "SageMaker prediction")
