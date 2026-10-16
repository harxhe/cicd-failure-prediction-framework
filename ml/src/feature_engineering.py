"""
M2 Feature Engineering Module

Transforms raw CI/CD build events (specs.md §4) into processed ML features
(specs.md §5).

DATA FLOW:
    Raw build events  →  validation  →  feature computation  →  processed features

FEATURES (specs.md §5, lines 94-101):
    files_changed                   — direct from raw field
    lines_changed                   — lines_added + lines_deleted
    test_coverage_delta             — PLACEHOLDER (raw data insufficient)
    author_recent_failure_rate      — trailing 30 days, cross-repo
    repo_recent_failure_rate        — trailing 30 days
    time_of_day_bucket              — from timestamp hour
    day_of_week                     — from timestamp
    prior_failures_last_5_builds    — previous 5 completed builds for same repo

DATA LEAKAGE PREVENTION:
    The current build's outcome is NEVER used when calculating its own
    prediction features. Historical features use only builds with timestamps
    strictly before the current build's timestamp.

ASSUMPTIONS (documented because specs.md does not define them):
    1. time_of_day_bucket hour boundaries:
       night=00:00-05:59, morning=06:00-11:59,
       afternoon=12:00-17:59, evening=18:00-23:59
    2. IN_PROGRESS builds are excluded from failure rate denominators
       (no final outcome to classify).
    3. author_recent_failure_rate is computed across ALL repos
       (the feature captures author reliability, not repo-specific).
    4. When fewer than 5 prior builds exist for prior_failures_last_5_builds,
       failures are counted from the available builds.
    5. Timestamps are parsed as UTC.
    6. Timestamp ties are broken by build_id (lexicographic) for determinism.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import sys
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from config.settings import (
    TIME_BUCKETS,
    DAY_OF_WEEK_NAMES,
    TRAILING_WINDOW_DAYS,
    PRIOR_BUILDS_COUNT,
    LABEL_MAP,
    OUTPUT_COLUMNS,
    TEST_COVERAGE_DATA_AVAILABLE,
    TEST_COVERAGE_PLACEHOLDER_VALUE,
)


# =============================================================================
# Individual Feature Functions
# =============================================================================

def compute_files_changed(raw_value):
    """
    Compute files_changed feature.

    Source:  raw files_changed (specs.md §4 L72)
    Output: integer or NaN
    """
    if pd.isna(raw_value):
        return np.nan
    return int(raw_value)


def compute_lines_changed(lines_added, lines_deleted):
    """
    Compute lines_changed feature.

    Definition: lines_added + lines_deleted (specs.md §5 L95)
    Output:     integer or NaN if EITHER source field is null.

    Rationale for NaN propagation: fabricating 0 when data is missing would
    undercount code churn, which is worse than an explicit missing value.
    """
    if pd.isna(lines_added) or pd.isna(lines_deleted):
        return np.nan
    return int(lines_added) + int(lines_deleted)


def compute_test_coverage_delta():
    """
    Compute test_coverage_delta feature.

    STATUS: **PLACEHOLDER** — returns a fixed value.

    REASON: The raw schema (specs.md §4) provides 'test_count' (integer)
    but NOT test coverage percentage. test_coverage_delta (specs.md §5 L96)
    is defined as a float in range [-1.0, 1.0] representing a coverage
    CHANGE. Computing this requires two coverage percentage readings
    (before and after), which are NOT available in the current raw data.

    test_count != test_coverage. These are fundamentally different metrics.

    DEPENDENCY: Requires M1 to add coverage percentage fields to the raw
    schema, or an external coverage reporting integration.

    Returns:
        float: TEST_COVERAGE_PLACEHOLDER_VALUE (0.0) — NOT real data.
    """
    return TEST_COVERAGE_PLACEHOLDER_VALUE


def compute_time_of_day_bucket(timestamp):
    """
    Compute time_of_day_bucket feature.

    Source:  timestamp hour (specs.md §5 L99)
    Output:  one of "morning" | "afternoon" | "evening" | "night"

    ASSUMPTION: Hour boundaries are NOT defined in specs.md.
    Using: night=0-5, morning=6-11, afternoon=12-17, evening=18-23.
    See config/settings.py for the authoritative boundary definition.
    """
    if pd.isna(timestamp):
        return None

    hour = timestamp.hour
    for bucket_name, (start_hour, end_hour) in TIME_BUCKETS.items():
        if start_hour <= hour <= end_hour:
            return bucket_name

    # Should never reach here if TIME_BUCKETS covers hours 0-23
    return None


def compute_day_of_week(timestamp):
    """
    Compute day_of_week feature.

    Source: timestamp (specs.md §5 L100)
    Output: one of "Mon" | "Tue" | "Wed" | "Thu" | "Fri" | "Sat" | "Sun"

    Uses Python's weekday() where Monday=0 → "Mon", Sunday=6 → "Sun",
    matching the exact strings in specs.md §5 L100.
    """
    if pd.isna(timestamp):
        return None
    return DAY_OF_WEEK_NAMES[timestamp.weekday()]


def compute_label(status):
    """
    Compute training label.

    Mapping (specs.md §5 L102-106):
        SUCCEEDED    → 0
        FAILED       → 1
        IN_PROGRESS  → None (null — "unlabeled/live build")

    Any unrecognized status maps to None.
    """
    if pd.isna(status):
        return None
    return LABEL_MAP.get(status, None)


# =============================================================================
# Historical Feature Functions
# =============================================================================

def compute_author_recent_failure_rate(history_df, author, current_timestamp):
    """
    Compute author_recent_failure_rate.

    Definition:  float 0.0-1.0, trailing 30 days (specs.md §5 L97)
    Scope:       All repos (ASSUMPTION — specs.md does not restrict to one repo)
    Window:      [current_timestamp - 30 days, current_timestamp)  (exclusive)
    Denominator: Only SUCCEEDED and FAILED builds (IN_PROGRESS excluded)
    Leakage:     Current build is excluded (strict < on timestamp)

    Returns 0.0 when:
    - No qualifying builds exist in the window
    - Author is null/unknown
    """
    if history_df.empty or pd.isna(author):
        return 0.0

    window_start = current_timestamp - timedelta(days=TRAILING_WINDOW_DAYS)

    mask = (
        (history_df["author"] == author)
        & (history_df["timestamp"] >= window_start)
        & (history_df["timestamp"] < current_timestamp)
        & (history_df["status"].isin(["SUCCEEDED", "FAILED"]))
    )
    author_builds = history_df.loc[mask]

    if len(author_builds) == 0:
        return 0.0

    failed_count = (author_builds["status"] == "FAILED").sum()
    return round(float(failed_count) / len(author_builds), 4)


def compute_repo_recent_failure_rate(history_df, repo, current_timestamp):
    """
    Compute repo_recent_failure_rate.

    Definition:  float 0.0-1.0, trailing 30 days (specs.md §5 L98)
    Window:      [current_timestamp - 30 days, current_timestamp)
    Denominator: Only SUCCEEDED and FAILED builds (IN_PROGRESS excluded)
    Leakage:     Current build is excluded (strict < on timestamp)

    Returns 0.0 when no qualifying builds exist.
    """
    if history_df.empty:
        return 0.0

    window_start = current_timestamp - timedelta(days=TRAILING_WINDOW_DAYS)

    mask = (
        (history_df["repo"] == repo)
        & (history_df["timestamp"] >= window_start)
        & (history_df["timestamp"] < current_timestamp)
        & (history_df["status"].isin(["SUCCEEDED", "FAILED"]))
    )
    repo_builds = history_df.loc[mask]

    if len(repo_builds) == 0:
        return 0.0

    failed_count = (repo_builds["status"] == "FAILED").sum()
    return round(float(failed_count) / len(repo_builds), 4)


def compute_prior_failures_last_5_builds(history_df, repo, current_timestamp):
    """
    Compute prior_failures_last_5_builds.

    Definition: integer 0-5 (specs.md §5 L101)
    Scope:      Previous 5 builds for the SAME repo
    Ordering:   By timestamp descending (specs.md §6 L123 confirms this pattern)
    Leakage:    Current build is excluded (strict < on timestamp)

    ASSUMPTION: IN_PROGRESS builds are excluded (they have no final outcome,
    so counting them would misrepresent the failure rate).

    When fewer than 5 prior completed builds exist, counts failures from
    the available builds (returning a value in 0..min(available, 5)).
    """
    if history_df.empty:
        return 0

    mask = (
        (history_df["repo"] == repo)
        & (history_df["timestamp"] < current_timestamp)
        & (history_df["status"].isin(["SUCCEEDED", "FAILED"]))
    )
    repo_builds = history_df.loc[mask].sort_values("timestamp", ascending=False)

    # Take the most recent 5 (or fewer if not enough history)
    recent = repo_builds.head(PRIOR_BUILDS_COUNT)

    if len(recent) == 0:
        return 0

    return int((recent["status"] == "FAILED").sum())


# =============================================================================
# Main Pipeline
# =============================================================================

def engineer_features(raw_df):
    """
    Main feature engineering pipeline.

    Transforms a DataFrame of raw build events (specs.md §4) into
    processed ML features (specs.md §5).

    Steps:
        1. Validate required fields exist.
        2. Parse timestamps and sort chronologically.
        3. For each build (in chronological order), compute all features
           using ONLY information from builds that occurred BEFORE it.
        4. Return processed DataFrame with exact column order from specs.md §5.

    Args:
        raw_df: pandas DataFrame with raw build events.

    Returns:
        tuple of (processed_df, pipeline_warnings):
            processed_df:     DataFrame with columns matching specs.md §5
            pipeline_warnings: list of human-readable warning/info strings
    """
    pipeline_warnings = []

    # ---- Step 1: Validate required fields ----
    required_fields = ["build_id", "repo", "timestamp", "status"]
    for field in required_fields:
        if field not in raw_df.columns:
            raise ValueError(
                f"Missing required field: '{field}' (specs.md §4). "
                f"Cannot proceed with feature engineering."
            )

    # ---- Step 2: Parse timestamps, sort chronologically ----
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    null_ts = int(df["timestamp"].isna().sum())
    if null_ts > 0:
        pipeline_warnings.append(
            f"WARNING: {null_ts} build(s) have unparseable/null timestamps. "
            f"Historical features will be degraded for these records."
        )

    # Deterministic sort: timestamp first, build_id as tiebreaker
    df = df.sort_values(["timestamp", "build_id"]).reset_index(drop=True)

    # ---- Step 2a: Validate status values ----
    if "status" in df.columns:
        invalid_mask = (
            ~df["status"].isin(["SUCCEEDED", "FAILED", "IN_PROGRESS"])
            & df["status"].notna()
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            invalid_vals = df.loc[invalid_mask, "status"].unique().tolist()
            pipeline_warnings.append(
                f"WARNING: {invalid_count} build(s) have invalid status "
                f"values: {invalid_vals}. These will produce label=None."
            )

    # ---- Step 2b: Null field warnings ----
    nullable_fields = {
        "files_changed": "files_changed feature will be NaN",
        "lines_added": "lines_changed feature will be NaN",
        "lines_deleted": "lines_changed feature will be NaN",
        "author": "author_recent_failure_rate will default to 0.0",
    }
    for field, consequence in nullable_fields.items():
        if field in df.columns:
            null_count = int(df[field].isna().sum())
            if null_count > 0:
                pipeline_warnings.append(
                    f"INFO: '{field}' has {null_count} null value(s) out of "
                    f"{len(df)} records. Consequence: {consequence}."
                )

    # ---- Step 2c: test_coverage_delta warning ----
    if not TEST_COVERAGE_DATA_AVAILABLE:
        pipeline_warnings.append(
            "WARNING [test_coverage_delta]: Set to PLACEHOLDER value "
            f"({TEST_COVERAGE_PLACEHOLDER_VALUE}) for ALL records. "
            "The raw schema (specs.md §4) provides 'test_count' (integer) "
            "but NOT test coverage percentage. This feature CANNOT be "
            "computed from current raw data. Value is NOT real coverage data. "
            "DEPENDENCY: M1 must add coverage percentage fields."
        )

    # ---- Step 3: Compute features for each build ----
    results = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        # History = all builds chronologically before this one.
        # This prevents future-data leakage.
        history = df.iloc[:idx]
        current_ts = row["timestamp"]

        record = {
            # Metadata (specs.md §5)
            "build_id": row["build_id"],
            "repo": row["repo"],

            # --- Feature: files_changed ---
            # Source: raw files_changed (specs.md §4 L72)
            "files_changed": compute_files_changed(
                row.get("files_changed", np.nan)
            ),

            # --- Feature: lines_changed ---
            # Definition: lines_added + lines_deleted (specs.md §5 L95)
            "lines_changed": compute_lines_changed(
                row.get("lines_added", np.nan),
                row.get("lines_deleted", np.nan),
            ),

            # --- Feature: test_coverage_delta ---
            # STATUS: PLACEHOLDER (see compute_test_coverage_delta docstring)
            "test_coverage_delta": compute_test_coverage_delta(),

            # --- Feature: author_recent_failure_rate ---
            # Trailing 30 days, cross-repo (specs.md §5 L97)
            "author_recent_failure_rate": compute_author_recent_failure_rate(
                history, row.get("author"), current_ts
            ),

            # --- Feature: repo_recent_failure_rate ---
            # Trailing 30 days (specs.md §5 L98)
            "repo_recent_failure_rate": compute_repo_recent_failure_rate(
                history, row["repo"], current_ts
            ),

            # --- Feature: time_of_day_bucket ---
            # From timestamp hour (specs.md §5 L99)
            "time_of_day_bucket": compute_time_of_day_bucket(current_ts),

            # --- Feature: day_of_week ---
            # From timestamp (specs.md §5 L100)
            "day_of_week": compute_day_of_week(current_ts),

            # --- Feature: prior_failures_last_5_builds ---
            # Previous 5 builds for same repo (specs.md §5 L101)
            "prior_failures_last_5_builds": compute_prior_failures_last_5_builds(
                history, row["repo"], current_ts
            ),

            # --- Label ---
            # 0=SUCCEEDED, 1=FAILED, null=IN_PROGRESS (specs.md §5 L102-106)
            "label": compute_label(row["status"]),
        }

        results.append(record)

    processed_df = pd.DataFrame(results)

    # ---- Step 4: Enforce exact column order from specs.md §5 ----
    processed_df = processed_df[OUTPUT_COLUMNS]

    return processed_df, pipeline_warnings


# =============================================================================
# Day 2 TravisTorrent pipeline
# =============================================================================

TRAVIS_SOURCE_COLUMNS = [
    "tr_build_id",
    "gh_project_name",
    "git_diff_src_churn",
    "git_diff_test_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "gh_test_lines_per_kloc",
    "gh_pushed_at",
    "gh_build_started_at",
    "tr_status",
    "tr_log_num_tests_ok",
    "tr_log_num_tests_failed",
    "tr_log_num_tests_run",
    "tr_log_num_tests_skipped",
    "tr_log_buildduration",
    "tr_duration",
]

TRAVIS_OUTPUT_COLUMNS = [
    "files_changed",
    "lines_changed",
    "repo_recent_failure_rate",
    "time_of_day_bucket",
    "day_of_week",
    "prior_failures_last_5_builds",
    "label",
]


def load_travistorrent(path, chunksize=100_000):
    """Load only Day 2 source columns from a compressed TravisTorrent CSV."""
    available = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    missing = [column for column in TRAVIS_SOURCE_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"Missing required TravisTorrent columns: {missing}")

    chunks = pd.read_csv(
        path,
        compression="gzip",
        usecols=TRAVIS_SOURCE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )
    return pd.concat(chunks, ignore_index=True)


def _normalize_travis_label(status):
    """Map observed TravisTorrent status values to 0/1, else missing."""
    value = str(status).strip().lower()
    if value in {"passed", "pass", "success", "succeeded"}:
        return 0
    if value in {"failed", "fail", "error", "errored"}:
        return 1
    return np.nan


def _travis_time_bucket(timestamp):
    if pd.isna(timestamp):
        return np.nan
    hour = timestamp.hour
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def engineer_travistorrent(raw_df):
    """Create Day 2 features using only earlier builds for historical values."""
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["gh_build_started_at"], utc=True, errors="coerce")
    df["label"] = df["tr_status"].map(_normalize_travis_label)
    df["repo"] = df["gh_project_name"]
    numeric_columns = [
        "git_diff_src_churn", "git_diff_test_churn",
        "gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["tr_build_id", "repo", "timestamp", "label"])
    df = df.sort_values(["timestamp", "tr_build_id"], kind="mergesort").reset_index(drop=True)
    files_changed = df[
        ["gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified"]
    ].sum(axis=1, min_count=1)
    lines_changed = df[["git_diff_src_churn", "git_diff_test_churn"]].sum(axis=1, min_count=1)
    repo_rates = np.zeros(len(df), dtype=float)
    prior_failures = np.zeros(len(df), dtype=int)

    # Vectorized per-repository searches keep only cumulative counts. For each
    # row, the left edge is the first row at the current timestamp, so ties do
    # not leak into one another and all future rows are excluded.
    for _, positions in df.groupby("repo", sort=False).groups.items():
        positions = np.asarray(positions)
        timestamps = df.loc[positions, "timestamp"].astype("int64").to_numpy()
        labels = df.loc[positions, "label"].to_numpy(dtype=int)
        current_start = timestamps - np.int64(timedelta(days=30).total_seconds() * 1_000_000_000)
        end = np.searchsorted(timestamps, timestamps, side="left")
        start = np.searchsorted(timestamps, current_start, side="left")
        cumulative = np.concatenate(([0], np.cumsum(labels)))
        window_count = end - start
        window_failures = cumulative[end] - cumulative[start]
        repo_rates[positions] = np.divide(
            window_failures,
            window_count,
            out=np.zeros_like(window_failures, dtype=float),
            where=window_count > 0,
        )
        prior_start = np.maximum(0, end - 5)
        prior_failures[positions] = cumulative[end] - cumulative[prior_start]

    result = pd.DataFrame({
        "files_changed": files_changed.to_numpy(),
        "lines_changed": lines_changed.to_numpy(),
        "repo_recent_failure_rate": repo_rates,
        "time_of_day_bucket": df["timestamp"].map(_travis_time_bucket).to_numpy(),
        "day_of_week": df["timestamp"].dt.day_name().str[:3].to_numpy(),
        "prior_failures_last_5_builds": prior_failures,
        "label": df["label"].astype(int).to_numpy(),
    })
    return result[TRAVIS_OUTPUT_COLUMNS]


def write_travistorrent_outputs(input_path, output_path, report_path):
    """Generate the Day 2 dataset and a reproducible EDA report."""
    raw = load_travistorrent(input_path)
    numeric = [
        "git_diff_src_churn", "git_diff_test_churn", "gh_diff_files_added",
        "gh_diff_files_deleted", "gh_diff_files_modified", "gh_test_lines_per_kloc",
        "tr_log_num_tests_ok", "tr_log_num_tests_failed", "tr_log_num_tests_run",
        "tr_log_num_tests_skipped", "tr_log_buildduration", "tr_duration",
    ]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    timestamps = pd.to_datetime(raw["gh_build_started_at"], utc=True, errors="coerce")
    labels = raw["tr_status"].map(_normalize_travis_label)
    output = engineer_travistorrent(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    stats = raw[numeric].describe().T.round(3).to_markdown()
    missing = raw[TRAVIS_SOURCE_COLUMNS].isna().sum().to_frame("count")
    missing["percent"] = (missing["count"] / len(raw) * 100).round(3)
    missing = missing.to_markdown()
    status_counts = labels.value_counts(dropna=False).to_dict()
    report = f"""# TravisTorrent Day 2 EDA Report

## Dataset

- Records read: **{len(raw):,}**
- Unique repositories: **{raw['gh_project_name'].nunique(dropna=True):,}**
- Build start range: **{timestamps.min()}** to **{timestamps.max()}**
- Rows retained for training features: **{len(output):,}**
- Rows removed: **{len(raw) - len(output):,}** (missing build ID, repository, timestamp, or recognized label)

## Labels

Raw normalized label counts: `{status_counts}`. The output contains:

```text
{output['label'].value_counts().sort_index().to_string()}
```

## Missing Values

Missing counts and percentages for selected source columns:

{missing}

Rows with missing required identity/timestamp/status values are removed. Numeric
source values remain missing and are represented as missing in derived sums;
historical repository rates default to `0.0` when no earlier completed builds
exist. No values are fabricated for unavailable coverage or author identity.

## Numerical Statistics

{stats}

## Observations and Limitations

- Source churn, test churn, changed-file counts, test-log fields, and duration
  were inspected from the relevant columns only.
- `test_coverage_delta` is unavailable: `gh_test_lines_per_kloc` measures test
  lines per KLOC, not coverage percentage or coverage change.
- `author_recent_failure_rate` is unavailable because no author identifier was
  selected or confirmed in the TravisTorrent schema.
- Repository history features use only rows earlier than the current build,
  after chronological sorting. The current label is never used in its own
  historical features.
- Feature variation must be checked from the generated output before training;
  constant or highly missing inputs should be excluded later.

## Final Day 2 Feature Set

`files_changed`, `lines_changed`, `repo_recent_failure_rate`,
`time_of_day_bucket`, `day_of_week`, `prior_failures_last_5_builds`, `label`.

No model was trained and no AWS resource was changed.
"""
    report_path.write_text(report, encoding="utf-8")
    return raw, output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Day 2 TravisTorrent features")
    parser.add_argument("--input", type=Path, default=_ML_ROOT / "data" / "travistorrent.csv.gz")
    parser.add_argument("--output", type=Path, default=_ML_ROOT / "data" / "training_features.csv")
    parser.add_argument("--report", type=Path, default=_ML_ROOT / "EDA_REPORT.md")
    args = parser.parse_args()
    write_travistorrent_outputs(args.input, args.output, args.report)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
