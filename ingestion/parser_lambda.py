"""
Parser Lambda: CodeBuild state-change event -> enriched JSON in S3 (raw/ prefix).

Environment variables:
  BUCKET_NAME    data lake bucket, e.g. cicdfp-data-lake-dev
  GITHUB_REPO    owner/name of the monitored repo, e.g. harxhe/cicd-failure-prediction-sample
  GITHUB_TOKEN   read-only token (optional for public repos, but avoids rate limits)
  PIPELINE_NAME  CodePipeline name (fallback for finding the commit SHA)
  STATE_MACHINE_ARN  M3 Step Functions workflow, started when a build begins

IAM (add to the Lambda role, besides s3:PutObject on raw/*):
  codebuild:BatchGetBuilds
  codepipeline:GetPipelineState
  states:StartExecution

Lambda timeout: raise to 20 seconds (the default 3 s is too short for the API calls).
"""
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
codebuild = boto3.client("codebuild")
codepipeline = boto3.client("codepipeline")
sfn = boto3.client("stepfunctions")

BUCKET_NAME = os.environ["BUCKET_NAME"]
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "cicdfp-sample-app-pipeline")
STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def build_start_time(event, info):
    """Real build start time (UTC, ISO-8601). Falls back to the event time."""
    raw = info.get("build-start-time")
    if raw:
        try:
            parsed = datetime.strptime(raw, "%b %d, %Y %I:%M:%S %p")
            return parsed.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            print(f"Could not parse build-start-time: {raw!r}")
    return event.get("time") or datetime.now(timezone.utc).isoformat()


def resolve_commit_sha(build_arn):
    """Return (sha, source). The CodeBuild event itself does not carry the SHA."""
    # 1) CodeBuild's own record of the resolved source version
    try:
        builds = codebuild.batch_get_builds(ids=[build_arn]).get("builds", [])
        if builds:
            version = builds[0].get("resolvedSourceVersion") or ""
            if SHA_RE.match(version):
                return version, "codebuild"
            print(f"resolvedSourceVersion is not a SHA: {version!r}")
    except Exception as exc:
        print(f"batch_get_builds failed: {exc}")

    # 2) Fallback: the revision the pipeline's source stage most recently saw
    try:
        state = codepipeline.get_pipeline_state(name=PIPELINE_NAME)
        for stage in state.get("stageStates", []):
            for action in stage.get("actionStates", []):
                revision = (action.get("currentRevision") or {}).get("revisionId", "")
                if SHA_RE.match(revision):
                    return revision, "codepipeline"
    except Exception as exc:
        print(f"get_pipeline_state failed: {exc}")

    return None, None


def fetch_commit_stats(sha):
    """Author and diff size for one commit, from the GitHub API."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{sha}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "cicdfp-parser"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        data = json.loads(response.read())

    stats = data.get("stats") or {}
    author = (data.get("commit", {}).get("author") or {}).get("name") or "unknown"
    return {
        "author": author,
        "files_changed": len(data.get("files") or []),
        "lines_added": int(stats.get("additions", 0)),
        "lines_deleted": int(stats.get("deletions", 0)),
    }


def start_prediction(build_id, repo):
    """Start the prediction workflow once per build, when the build begins.

    The execution name is the build id, so a duplicate EventBridge delivery
    cannot start a second run (and send a second alert).
    """
    if not STATE_MACHINE_ARN:
        return "skipped: STATE_MACHINE_ARN not set"
    try:
        sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=build_id,
            input=json.dumps({"build_id": build_id, "repo": repo}),
        )
        return "started"
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ExecutionAlreadyExists":
            return "already started"
        print(f"start_execution failed: {exc}")
        return f"failed: {exc}"
    except Exception as exc:
        print(f"start_execution failed: {exc}")
        return f"failed: {exc}"


def lambda_handler(event, context):
    detail = event.get("detail", {})
    info = detail.get("additional-information", {}) or {}

    build_arn = detail.get("build-id", "")
    build_id = build_arn.split(":")[-1] if build_arn else "unknown"
    project_name = detail.get("project-name", "unknown-repo")
    status = detail.get("build-status", "UNKNOWN")
    phases = info.get("phases", []) or []

    failed_stage = next(
        (p.get("phase-type") for p in phases if p.get("phase-status") == "FAILED"), None
    )
    duration = None
    if info.get("build-complete"):
        duration = float(sum(p.get("duration-in-seconds", 0) for p in phases))

    commit_sha, sha_source = resolve_commit_sha(build_arn)

    # Numeric fields default to 0, never null: the M2 ETL calls int() on them.
    stats = {"author": "unknown", "files_changed": 0, "lines_added": 0, "lines_deleted": 0}
    if not commit_sha:
        enrichment = "failed: no commit sha"
    elif not GITHUB_REPO:
        enrichment = "failed: GITHUB_REPO not set"
    else:
        try:
            stats = fetch_commit_stats(commit_sha)
            enrichment = "ok"
        except Exception as exc:
            enrichment = f"failed: {exc}"

    record = {
        "build_id": build_id,
        "repo": project_name,
        "branch": "main",
        "commit_sha": commit_sha or "unknown",
        "author": stats["author"],
        "timestamp": build_start_time(event, info),
        "files_changed": stats["files_changed"],
        "lines_added": stats["lines_added"],
        "lines_deleted": stats["lines_deleted"],
        "test_count": 0,
        "build_duration_seconds": duration,
        "status": status,
        "failed_stage": failed_stage,
        "sha_source": sha_source,
        "enrichment_status": enrichment,
    }

    key = f"raw/{project_name}/{build_id}.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(record),
        ContentType="application/json",
    )

    # Predict at build start; the final status only becomes the training label.
    workflow = "not applicable"
    if status == "IN_PROGRESS":
        workflow = start_prediction(build_id, project_name)

    print(json.dumps({"wrote": key, "status": status, "sha_source": sha_source,
                      "enrichment": enrichment, "workflow": workflow}))
    return {"statusCode": 200, "body": f"Wrote {key}"}