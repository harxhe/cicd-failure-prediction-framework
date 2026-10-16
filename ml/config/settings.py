"""
M2 ML & Prediction Layer — Configuration Settings

ALL values in this file are sourced from specs.md and PROJECT_PLAN.md.
DO NOT modify shared names/contracts without team sync (specs.md §12).

Source references are provided as inline comments.
"""

# =============================================================================
# AWS Configuration (specs.md §11, lines 209-217)
# =============================================================================
AWS_REGION = "ap-south-1"
S3_BUCKET = "cicdfp-data-lake-dev"
S3_RAW_PREFIX = "raw/"                            # specs.md §3 L46
S3_PROCESSED_PREFIX = "processed/"                # specs.md §3 L48
S3_PROCESSED_FILE = "processed/features.parquet"  # specs.md §3 L48
S3_MODELS_PREFIX = "models/"                      # specs.md §3 L50
DYNAMODB_TABLE = "m2-build-features-table"        # specs.md §6 L112, §11 L213
SAGEMAKER_ENDPOINT = "m2-xgboost-endpoint-v2-dev"
SAGEMAKER_MODEL = "m2-xgboost-model-v2-dev"

# =============================================================================
# Risk Thresholds (specs.md §7 L157 — fixed, do not change without team sync)
# =============================================================================
RISK_THRESHOLD_LOW = 0.4    # risk_score < 0.4 → "low"
RISK_THRESHOLD_HIGH = 0.7   # risk_score > 0.7 → "high"
                            # 0.4 ≤ risk_score ≤ 0.7 → "medium"

# =============================================================================
# Raw Build Event Schema (specs.md §4, lines 64-79)
# =============================================================================
RAW_SCHEMA_FIELDS = [
    "build_id",               # string — unique ID from CodeBuild
    "repo",                   # string — repository name
    "branch",                 # string
    "commit_sha",             # string
    "author",                 # string
    "timestamp",              # ISO-8601 string — build start time
    "files_changed",          # integer
    "lines_added",            # integer
    "lines_deleted",          # integer
    "test_count",             # integer
    "build_duration_seconds", # integer, null if build still running
    "status",                 # SUCCEEDED | FAILED | IN_PROGRESS
    "failed_stage",           # string | null
]

# Valid status values (specs.md §4 L77)
VALID_STATUSES = ["SUCCEEDED", "FAILED", "IN_PROGRESS"]

# =============================================================================
# Processed Feature Schema (specs.md §5, lines 90-103)
# =============================================================================
FEATURE_COLUMNS = [
    "files_changed",                  # int     — direct from raw
    "lines_changed",                  # int     — lines_added + lines_deleted
    "test_coverage_delta",            # float   — PLACEHOLDER (see below)
    "author_recent_failure_rate",     # float   — trailing 30 days
    "repo_recent_failure_rate",       # float   — trailing 30 days
    "time_of_day_bucket",            # string  — morning|afternoon|evening|night
    "day_of_week",                   # string  — Mon|Tue|…|Sun
    "prior_failures_last_5_builds",  # int 0-5 — previous 5 repo builds
]

# Full output column order (specs.md §5)
OUTPUT_COLUMNS = ["build_id", "repo"] + FEATURE_COLUMNS + ["label"]

# =============================================================================
# Label Mapping (specs.md §5 L102-106)
#   1 = failed
#   0 = succeeded
#   null = unlabeled / live build (IN_PROGRESS)
# =============================================================================
LABEL_MAP = {
    "SUCCEEDED": 0,
    "FAILED": 1,
    "IN_PROGRESS": None,  # "null for a live build being scored" — specs.md §5 L106
}

# =============================================================================
# Time-of-Day Buckets
#
# specs.md §5 L99 defines the four bucket NAMES:
#     morning | afternoon | evening | night
#
# ASSUMPTION: The hour boundaries below are NOT defined in specs.md.
# These are implementation assumptions documented per project requirement:
#     night     = 00:00–05:59  (hours 0-5)
#     morning   = 06:00–11:59  (hours 6-11)
#     afternoon = 12:00–17:59  (hours 12-17)
#     evening   = 18:00–23:59  (hours 18-23)
# =============================================================================
TIME_BUCKETS = {
    "night":     (0, 5),
    "morning":   (6, 11),
    "afternoon": (12, 17),
    "evening":   (18, 23),
}

# =============================================================================
# Day-of-Week Values (specs.md §5 L100 — exact strings)
# =============================================================================
DAY_OF_WEEK_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# =============================================================================
# Historical Feature Parameters (specs.md §5 L97-98, L101)
# =============================================================================
TRAILING_WINDOW_DAYS = 30   # "trailing 30 days" — specs.md §5 L97-98
PRIOR_BUILDS_COUNT = 5      # "last 5 builds" — specs.md §5 L101

# =============================================================================
# test_coverage_delta — Data Availability Flag
#
# The raw schema (specs.md §4) provides "test_count" (integer) but does NOT
# provide test coverage percentage. test_coverage_delta (specs.md §5 L96)
# requires coverage percentage readings, which are unavailable.
#
# This flag indicates that coverage data is a PLACEHOLDER, not real data.
# Downstream code (training, inference, reporting) should check this flag
# and treat the feature accordingly.
# =============================================================================
TEST_COVERAGE_DATA_AVAILABLE = False
TEST_COVERAGE_PLACEHOLDER_VALUE = 0.0
