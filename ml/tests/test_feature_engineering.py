"""
M2 Feature Engineering Tests

Comprehensive tests for every M2 transformation, covering:
    1.  lines_changed calculation
    2.  timestamp parsing
    3.  time_of_day_bucket (all 4 buckets + boundaries)
    4.  day_of_week (all 7 days)
    5.  author_recent_failure_rate (trailing 30 days)
    6.  repo_recent_failure_rate (trailing 30 days)
    7.  prior_failures_last_5_builds
    8.  label generation
    9.  missing/null values
    10. future-data leakage prevention
    11. fewer than 5 previous builds
    12. multiple repositories
    13. multiple authors
    14. schema validation (data_loader)
    15. test_coverage_delta placeholder

All test data is deterministic — no randomness.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Path setup
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from src.feature_engineering import (
    compute_files_changed,
    compute_lines_changed,
    compute_test_coverage_delta,
    compute_time_of_day_bucket,
    compute_day_of_week,
    compute_label,
    compute_author_recent_failure_rate,
    compute_repo_recent_failure_rate,
    compute_prior_failures_last_5_builds,
    engineer_features,
)
from src.data_loader import validate_raw_schema, load_raw_builds_json
from config.settings import TEST_COVERAGE_PLACEHOLDER_VALUE, OUTPUT_COLUMNS


# =============================================================================
# Helper: Build a small DataFrame for historical feature tests
# =============================================================================

def _make_history_df(records):
    """Create a DataFrame from a list of dicts, parsing timestamps."""
    if not records:
        return pd.DataFrame(columns=["author", "timestamp", "repo", "status"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# =============================================================================
# 1. files_changed
# =============================================================================

class TestComputeFilesChanged:
    def test_normal_value(self):
        assert compute_files_changed(5) == 5

    def test_zero(self):
        assert compute_files_changed(0) == 0

    def test_null(self):
        assert np.isnan(compute_files_changed(np.nan))

    def test_none(self):
        assert np.isnan(compute_files_changed(None))


# =============================================================================
# 2. lines_changed (lines_added + lines_deleted, specs.md §5 L95)
# =============================================================================

class TestComputeLinesChanged:
    def test_both_present(self):
        assert compute_lines_changed(120, 30) == 150

    def test_both_zero(self):
        assert compute_lines_changed(0, 0) == 0

    def test_added_null(self):
        assert np.isnan(compute_lines_changed(None, 30))

    def test_deleted_null(self):
        assert np.isnan(compute_lines_changed(120, None))

    def test_both_null(self):
        assert np.isnan(compute_lines_changed(None, None))

    def test_large_values(self):
        assert compute_lines_changed(10000, 5000) == 15000


# =============================================================================
# 3. test_coverage_delta (PLACEHOLDER)
# =============================================================================

class TestComputeTestCoverageDelta:
    def test_returns_placeholder_value(self):
        result = compute_test_coverage_delta()
        assert result == TEST_COVERAGE_PLACEHOLDER_VALUE

    def test_returns_float(self):
        result = compute_test_coverage_delta()
        assert isinstance(result, float)

    def test_within_spec_range(self):
        """specs.md §5 L96: range [-1.0, 1.0]"""
        result = compute_test_coverage_delta()
        assert -1.0 <= result <= 1.0


# =============================================================================
# 4. time_of_day_bucket
# =============================================================================

class TestComputeTimeOfDayBucket:
    """
    ASSUMPTION boundaries (not defined in specs.md):
        night:     00:00-05:59
        morning:   06:00-11:59
        afternoon: 12:00-17:59
        evening:   18:00-23:59
    """
    def test_night_midnight(self):
        ts = pd.Timestamp("2026-08-01T00:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "night"

    def test_night_3am(self):
        ts = pd.Timestamp("2026-08-01T03:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "night"

    def test_night_boundary_5am(self):
        ts = pd.Timestamp("2026-08-01T05:59:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "night"

    def test_morning_boundary_6am(self):
        ts = pd.Timestamp("2026-08-01T06:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "morning"

    def test_morning_9am(self):
        ts = pd.Timestamp("2026-08-01T09:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "morning"

    def test_morning_boundary_11am(self):
        ts = pd.Timestamp("2026-08-01T11:59:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "morning"

    def test_afternoon_boundary_noon(self):
        ts = pd.Timestamp("2026-08-01T12:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "afternoon"

    def test_afternoon_3pm(self):
        ts = pd.Timestamp("2026-08-01T15:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "afternoon"

    def test_afternoon_boundary_5pm(self):
        ts = pd.Timestamp("2026-08-01T17:59:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "afternoon"

    def test_evening_boundary_6pm(self):
        ts = pd.Timestamp("2026-08-01T18:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "evening"

    def test_evening_9pm(self):
        ts = pd.Timestamp("2026-08-01T21:00:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "evening"

    def test_evening_boundary_11pm(self):
        ts = pd.Timestamp("2026-08-01T23:59:00", tz="UTC")
        assert compute_time_of_day_bucket(ts) == "evening"

    def test_null_timestamp(self):
        assert compute_time_of_day_bucket(pd.NaT) is None


# =============================================================================
# 5. day_of_week
# =============================================================================

class TestComputeDayOfWeek:
    """Verify all 7 day names match specs.md §5 L100."""

    def test_monday(self):
        # 2026-08-03 is a Monday
        ts = pd.Timestamp("2026-08-03", tz="UTC")
        assert compute_day_of_week(ts) == "Mon"

    def test_tuesday(self):
        ts = pd.Timestamp("2026-08-04", tz="UTC")
        assert compute_day_of_week(ts) == "Tue"

    def test_wednesday(self):
        ts = pd.Timestamp("2026-08-05", tz="UTC")
        assert compute_day_of_week(ts) == "Wed"

    def test_thursday(self):
        ts = pd.Timestamp("2026-08-06", tz="UTC")
        assert compute_day_of_week(ts) == "Thu"

    def test_friday(self):
        ts = pd.Timestamp("2026-08-07", tz="UTC")
        assert compute_day_of_week(ts) == "Fri"

    def test_saturday(self):
        ts = pd.Timestamp("2026-08-08", tz="UTC")
        assert compute_day_of_week(ts) == "Sat"

    def test_sunday(self):
        ts = pd.Timestamp("2026-08-09", tz="UTC")
        assert compute_day_of_week(ts) == "Sun"

    def test_null_timestamp(self):
        assert compute_day_of_week(pd.NaT) is None


# =============================================================================
# 6. label generation (specs.md §5 L102-106)
# =============================================================================

class TestComputeLabel:
    def test_succeeded(self):
        assert compute_label("SUCCEEDED") == 0

    def test_failed(self):
        assert compute_label("FAILED") == 1

    def test_in_progress(self):
        assert compute_label("IN_PROGRESS") is None

    def test_null_status(self):
        assert compute_label(None) is None

    def test_invalid_status(self):
        assert compute_label("CANCELLED") is None


# =============================================================================
# 7. author_recent_failure_rate (trailing 30 days, specs.md §5 L97)
# =============================================================================

class TestAuthorRecentFailureRate:
    def test_with_mixed_history(self):
        """Alice has 1 FAILED out of 3 builds in 30-day window → 1/3."""
        history = _make_history_df([
            {"author": "alice", "timestamp": "2026-08-01T10:00:00Z",
             "repo": "r1", "status": "SUCCEEDED"},
            {"author": "alice", "timestamp": "2026-08-05T10:00:00Z",
             "repo": "r1", "status": "FAILED"},
            {"author": "alice", "timestamp": "2026-08-10T10:00:00Z",
             "repo": "r2", "status": "SUCCEEDED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "alice", current_ts)
        assert abs(rate - 1 / 3) < 0.001

    def test_no_history(self):
        history = _make_history_df([])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "alice", current_ts)
        assert rate == 0.0

    def test_null_author(self):
        history = _make_history_df([
            {"author": "alice", "timestamp": "2026-08-01T10:00:00Z",
             "repo": "r1", "status": "FAILED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, None, current_ts)
        assert rate == 0.0

    def test_outside_30_day_window(self):
        """Build from 40 days ago should NOT be counted."""
        history = _make_history_df([
            {"author": "alice", "timestamp": "2026-07-01T10:00:00Z",
             "repo": "r1", "status": "FAILED"},
        ])
        # Current build is Aug 15 → window starts Jul 16
        # Jul 1 is outside the window
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "alice", current_ts)
        assert rate == 0.0

    def test_all_failures(self):
        history = _make_history_df([
            {"author": "bob", "timestamp": "2026-08-01T10:00:00Z",
             "repo": "r1", "status": "FAILED"},
            {"author": "bob", "timestamp": "2026-08-05T10:00:00Z",
             "repo": "r1", "status": "FAILED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "bob", current_ts)
        assert rate == 1.0

    def test_no_failures(self):
        history = _make_history_df([
            {"author": "bob", "timestamp": "2026-08-01T10:00:00Z",
             "repo": "r1", "status": "SUCCEEDED"},
            {"author": "bob", "timestamp": "2026-08-05T10:00:00Z",
             "repo": "r1", "status": "SUCCEEDED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "bob", current_ts)
        assert rate == 0.0

    def test_excludes_in_progress(self):
        """IN_PROGRESS builds should not count in numerator or denominator."""
        history = _make_history_df([
            {"author": "alice", "timestamp": "2026-08-01T10:00:00Z",
             "repo": "r1", "status": "SUCCEEDED"},
            {"author": "alice", "timestamp": "2026-08-05T10:00:00Z",
             "repo": "r1", "status": "IN_PROGRESS"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "alice", current_ts)
        # Only 1 qualifying build (SUCCEEDED), so rate = 0/1 = 0.0
        assert rate == 0.0

    def test_cross_repo(self):
        """Author failure rate spans all repos (ASSUMPTION)."""
        history = _make_history_df([
            {"author": "alice", "timestamp": "2026-08-01T10:00:00Z",
             "repo": "repo-a", "status": "FAILED"},
            {"author": "alice", "timestamp": "2026-08-05T10:00:00Z",
             "repo": "repo-b", "status": "SUCCEEDED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "alice", current_ts)
        # 1 FAILED out of 2 total → 0.5
        assert rate == 0.5

    def test_excludes_other_authors(self):
        """Only the specified author's builds count."""
        history = _make_history_df([
            {"author": "alice", "timestamp": "2026-08-01T10:00:00Z",
             "repo": "r1", "status": "SUCCEEDED"},
            {"author": "bob", "timestamp": "2026-08-05T10:00:00Z",
             "repo": "r1", "status": "FAILED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_author_recent_failure_rate(history, "alice", current_ts)
        assert rate == 0.0  # alice's only build was SUCCEEDED


# =============================================================================
# 8. repo_recent_failure_rate (trailing 30 days, specs.md §5 L98)
# =============================================================================

class TestRepoRecentFailureRate:
    def test_with_mixed_history(self):
        """2 FAILED out of 4 builds → 0.5."""
        history = _make_history_df([
            {"repo": "repo-a", "timestamp": "2026-08-01T10:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "repo-a", "timestamp": "2026-08-03T10:00:00Z",
             "author": "b", "status": "FAILED"},
            {"repo": "repo-a", "timestamp": "2026-08-05T10:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "repo-a", "timestamp": "2026-08-07T10:00:00Z",
             "author": "b", "status": "SUCCEEDED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_repo_recent_failure_rate(history, "repo-a", current_ts)
        assert rate == 0.5

    def test_no_history(self):
        history = _make_history_df([])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_repo_recent_failure_rate(history, "repo-a", current_ts)
        assert rate == 0.0

    def test_outside_window(self):
        history = _make_history_df([
            {"repo": "repo-a", "timestamp": "2026-07-01T10:00:00Z",
             "author": "a", "status": "FAILED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_repo_recent_failure_rate(history, "repo-a", current_ts)
        assert rate == 0.0

    def test_excludes_other_repos(self):
        history = _make_history_df([
            {"repo": "repo-a", "timestamp": "2026-08-01T10:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "repo-b", "timestamp": "2026-08-02T10:00:00Z",
             "author": "a", "status": "FAILED"},
        ])
        current_ts = pd.Timestamp("2026-08-15T10:00:00Z")
        rate = compute_repo_recent_failure_rate(history, "repo-a", current_ts)
        assert rate == 0.0


# =============================================================================
# 9. prior_failures_last_5_builds (specs.md §5 L101)
# =============================================================================

class TestPriorFailuresLast5Builds:
    def test_exactly_5_prior(self):
        """5 prior builds: S, F, F, S, F → 3 failures."""
        history = _make_history_df([
            {"repo": "r", "timestamp": "2026-08-01T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "r", "timestamp": "2026-08-02T01:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "r", "timestamp": "2026-08-03T01:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "r", "timestamp": "2026-08-04T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "r", "timestamp": "2026-08-05T01:00:00Z",
             "author": "a", "status": "FAILED"},
        ])
        current_ts = pd.Timestamp("2026-08-10T10:00:00Z")
        count = compute_prior_failures_last_5_builds(history, "r", current_ts)
        assert count == 3

    def test_more_than_5_prior(self):
        """7 prior builds — should only look at most recent 5."""
        history = _make_history_df([
            # These 2 are oldest → should NOT be counted
            {"repo": "r", "timestamp": "2026-08-01T01:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "r", "timestamp": "2026-08-02T01:00:00Z",
             "author": "a", "status": "FAILED"},
            # These 5 are the most recent → counted
            {"repo": "r", "timestamp": "2026-08-03T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "r", "timestamp": "2026-08-04T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "r", "timestamp": "2026-08-05T01:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "r", "timestamp": "2026-08-06T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "r", "timestamp": "2026-08-07T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
        ])
        current_ts = pd.Timestamp("2026-08-10T10:00:00Z")
        count = compute_prior_failures_last_5_builds(history, "r", current_ts)
        # Most recent 5: Aug7(S), Aug6(S), Aug5(F), Aug4(S), Aug3(S) → 1
        assert count == 1

    def test_fewer_than_5_prior(self):
        """Only 2 prior builds: S, F → 1 failure."""
        history = _make_history_df([
            {"repo": "r", "timestamp": "2026-08-01T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
            {"repo": "r", "timestamp": "2026-08-02T01:00:00Z",
             "author": "a", "status": "FAILED"},
        ])
        current_ts = pd.Timestamp("2026-08-10T10:00:00Z")
        count = compute_prior_failures_last_5_builds(history, "r", current_ts)
        assert count == 1

    def test_no_prior_builds(self):
        history = _make_history_df([])
        current_ts = pd.Timestamp("2026-08-10T10:00:00Z")
        count = compute_prior_failures_last_5_builds(history, "r", current_ts)
        assert count == 0

    def test_all_failures(self):
        history = _make_history_df([
            {"repo": "r", "timestamp": f"2026-08-0{i}T01:00:00Z",
             "author": "a", "status": "FAILED"}
            for i in range(1, 6)
        ])
        current_ts = pd.Timestamp("2026-08-10T10:00:00Z")
        count = compute_prior_failures_last_5_builds(history, "r", current_ts)
        assert count == 5

    def test_excludes_other_repos(self):
        history = _make_history_df([
            {"repo": "repo-a", "timestamp": "2026-08-01T01:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "repo-b", "timestamp": "2026-08-02T01:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "repo-a", "timestamp": "2026-08-03T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
        ])
        current_ts = pd.Timestamp("2026-08-10T10:00:00Z")
        count = compute_prior_failures_last_5_builds(
            history, "repo-a", current_ts
        )
        # repo-a has 2 builds: Aug1(F), Aug3(S) → 1 failure
        assert count == 1

    def test_excludes_in_progress(self):
        """IN_PROGRESS should not be counted (no final outcome)."""
        history = _make_history_df([
            {"repo": "r", "timestamp": "2026-08-01T01:00:00Z",
             "author": "a", "status": "FAILED"},
            {"repo": "r", "timestamp": "2026-08-02T01:00:00Z",
             "author": "a", "status": "IN_PROGRESS"},
            {"repo": "r", "timestamp": "2026-08-03T01:00:00Z",
             "author": "a", "status": "SUCCEEDED"},
        ])
        current_ts = pd.Timestamp("2026-08-10T10:00:00Z")
        count = compute_prior_failures_last_5_builds(history, "r", current_ts)
        # Only 2 completed builds: Aug1(F), Aug3(S) → 1 failure
        assert count == 1


# =============================================================================
# 10. Pipeline integration tests
# =============================================================================

class TestEngineerFeatures:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def small_raw_df(self):
        """Minimal raw DataFrame for pipeline tests."""
        return pd.DataFrame([
            {
                "build_id": "b1", "repo": "r1", "branch": "main",
                "commit_sha": "abc", "author": "alice",
                "timestamp": "2026-08-01T09:00:00Z",
                "files_changed": 5, "lines_added": 100, "lines_deleted": 20,
                "test_count": 10, "build_duration_seconds": 120,
                "status": "SUCCEEDED", "failed_stage": None,
            },
            {
                "build_id": "b2", "repo": "r1", "branch": "feature/x",
                "commit_sha": "def", "author": "bob",
                "timestamp": "2026-08-03T14:30:00Z",
                "files_changed": 10, "lines_added": 200, "lines_deleted": 50,
                "test_count": 12, "build_duration_seconds": 180,
                "status": "FAILED", "failed_stage": "test",
            },
            {
                "build_id": "b3", "repo": "r1", "branch": "main",
                "commit_sha": "ghi", "author": "alice",
                "timestamp": "2026-08-05T20:00:00Z",
                "files_changed": 3, "lines_added": 50, "lines_deleted": 10,
                "test_count": 12, "build_duration_seconds": 100,
                "status": "SUCCEEDED", "failed_stage": None,
            },
        ])

    def test_output_columns_match_spec(self, small_raw_df):
        """Output columns must match specs.md §5 exactly."""
        processed, _ = engineer_features(small_raw_df)
        assert list(processed.columns) == OUTPUT_COLUMNS

    def test_no_future_leakage_first_build(self, small_raw_df):
        """First build (chronologically) must have 0 rates and 0 prior failures."""
        processed, _ = engineer_features(small_raw_df)
        first = processed.iloc[0]
        assert first["author_recent_failure_rate"] == 0.0
        assert first["repo_recent_failure_rate"] == 0.0
        assert first["prior_failures_last_5_builds"] == 0

    def test_no_future_leakage_second_build(self, small_raw_df):
        """Second build should only see the first build's outcome."""
        processed, _ = engineer_features(small_raw_df)
        second = processed.iloc[1]
        # b2 is by bob, b1 was by alice → bob has no history → 0.0
        assert second["author_recent_failure_rate"] == 0.0
        # repo r1: only b1 (SUCCEEDED) before b2 → 0/1 = 0.0
        assert second["repo_recent_failure_rate"] == 0.0
        # prior 5 builds for r1: only b1 (SUCCEEDED) → 0 failures
        assert second["prior_failures_last_5_builds"] == 0

    def test_third_build_sees_first_two(self, small_raw_df):
        """Third build should see both first and second build outcomes."""
        processed, _ = engineer_features(small_raw_df)
        third = processed.iloc[2]
        # b3 is by alice. alice's prior: b1 (SUCCEEDED) → 0/1 = 0.0
        assert third["author_recent_failure_rate"] == 0.0
        # repo r1: b1(S), b2(F) → 1/2 = 0.5
        assert third["repo_recent_failure_rate"] == 0.5
        # prior failures for r1: b2(F), b1(S) → 1
        assert third["prior_failures_last_5_builds"] == 1

    def test_lines_changed_values(self, small_raw_df):
        processed, _ = engineer_features(small_raw_df)
        assert processed.iloc[0]["lines_changed"] == 120  # 100 + 20
        assert processed.iloc[1]["lines_changed"] == 250  # 200 + 50
        assert processed.iloc[2]["lines_changed"] == 60   # 50 + 10

    def test_label_values(self, small_raw_df):
        processed, _ = engineer_features(small_raw_df)
        assert processed.iloc[0]["label"] == 0  # SUCCEEDED
        assert processed.iloc[1]["label"] == 1  # FAILED
        assert processed.iloc[2]["label"] == 0  # SUCCEEDED

    def test_time_of_day_bucket_values(self, small_raw_df):
        processed, _ = engineer_features(small_raw_df)
        assert processed.iloc[0]["time_of_day_bucket"] == "morning"    # 09:00
        assert processed.iloc[1]["time_of_day_bucket"] == "afternoon"  # 14:30
        assert processed.iloc[2]["time_of_day_bucket"] == "evening"    # 20:00

    def test_day_of_week_values(self, small_raw_df):
        processed, _ = engineer_features(small_raw_df)
        # 2026-08-01 = Saturday, 2026-08-03 = Monday, 2026-08-05 = Wednesday
        assert processed.iloc[0]["day_of_week"] == "Sat"
        assert processed.iloc[1]["day_of_week"] == "Mon"
        assert processed.iloc[2]["day_of_week"] == "Wed"

    def test_test_coverage_delta_is_placeholder(self, small_raw_df):
        """All records must have the placeholder value."""
        processed, warnings_list = engineer_features(small_raw_df)
        assert (processed["test_coverage_delta"] == TEST_COVERAGE_PLACEHOLDER_VALUE).all()
        # Pipeline must emit a warning about this
        coverage_warnings = [
            w for w in warnings_list if "test_coverage_delta" in w
        ]
        assert len(coverage_warnings) > 0
        assert "PLACEHOLDER" in coverage_warnings[0]

    def test_chronological_ordering(self):
        """Builds should be processed in timestamp order, not input order."""
        raw = pd.DataFrame([
            {
                "build_id": "late", "repo": "r", "branch": "main",
                "commit_sha": "x", "author": "a",
                "timestamp": "2026-08-10T10:00:00Z",
                "files_changed": 1, "lines_added": 10, "lines_deleted": 5,
                "test_count": 5, "build_duration_seconds": 60,
                "status": "FAILED", "failed_stage": "test",
            },
            {
                "build_id": "early", "repo": "r", "branch": "main",
                "commit_sha": "y", "author": "a",
                "timestamp": "2026-08-01T10:00:00Z",
                "files_changed": 2, "lines_added": 20, "lines_deleted": 10,
                "test_count": 5, "build_duration_seconds": 90,
                "status": "SUCCEEDED", "failed_stage": None,
            },
        ])
        processed, _ = engineer_features(raw)
        # "early" should appear first after sorting
        assert processed.iloc[0]["build_id"] == "early"
        assert processed.iloc[1]["build_id"] == "late"
        # "late" should see "early" in its history
        assert processed.iloc[1]["prior_failures_last_5_builds"] == 0

    def test_in_progress_label_is_none(self):
        """IN_PROGRESS builds should have label=None."""
        raw = pd.DataFrame([
            {
                "build_id": "ip1", "repo": "r", "branch": "main",
                "commit_sha": "x", "author": "a",
                "timestamp": "2026-08-01T10:00:00Z",
                "files_changed": 1, "lines_added": 10, "lines_deleted": 5,
                "test_count": 5, "build_duration_seconds": None,
                "status": "IN_PROGRESS", "failed_stage": None,
            },
        ])
        processed, _ = engineer_features(raw)
        assert processed.iloc[0]["label"] is None

    def test_null_fields_propagate(self):
        """Null files_changed and lines_added/deleted → NaN in output."""
        raw = pd.DataFrame([
            {
                "build_id": "n1", "repo": "r", "branch": "main",
                "commit_sha": "x", "author": "a",
                "timestamp": "2026-08-01T10:00:00Z",
                "files_changed": None, "lines_added": None,
                "lines_deleted": None, "test_count": None,
                "build_duration_seconds": None,
                "status": "SUCCEEDED", "failed_stage": None,
            },
        ])
        processed, _ = engineer_features(raw)
        assert np.isnan(processed.iloc[0]["files_changed"])
        assert np.isnan(processed.iloc[0]["lines_changed"])

    def test_multiple_repos_isolated(self):
        """Features for repo-a should not be affected by repo-b builds."""
        raw = pd.DataFrame([
            {
                "build_id": "a1", "repo": "repo-a", "branch": "main",
                "commit_sha": "x", "author": "alice",
                "timestamp": "2026-08-01T10:00:00Z",
                "files_changed": 1, "lines_added": 10, "lines_deleted": 5,
                "test_count": 5, "build_duration_seconds": 60,
                "status": "FAILED", "failed_stage": "test",
            },
            {
                "build_id": "b1", "repo": "repo-b", "branch": "main",
                "commit_sha": "y", "author": "bob",
                "timestamp": "2026-08-02T10:00:00Z",
                "files_changed": 2, "lines_added": 20, "lines_deleted": 10,
                "test_count": 5, "build_duration_seconds": 90,
                "status": "SUCCEEDED", "failed_stage": None,
            },
            {
                "build_id": "a2", "repo": "repo-a", "branch": "main",
                "commit_sha": "z", "author": "alice",
                "timestamp": "2026-08-03T10:00:00Z",
                "files_changed": 3, "lines_added": 30, "lines_deleted": 15,
                "test_count": 5, "build_duration_seconds": 70,
                "status": "SUCCEEDED", "failed_stage": None,
            },
        ])
        processed, _ = engineer_features(raw)

        # a2 (repo-a): repo rate = 1/1 (a1 was FAILED) = 1.0
        a2 = processed[processed["build_id"] == "a2"].iloc[0]
        assert a2["repo_recent_failure_rate"] == 1.0
        assert a2["prior_failures_last_5_builds"] == 1

        # b1 (repo-b): no prior builds for repo-b → 0.0
        b1 = processed[processed["build_id"] == "b1"].iloc[0]
        assert b1["repo_recent_failure_rate"] == 0.0
        assert b1["prior_failures_last_5_builds"] == 0

    def test_missing_required_field_raises(self):
        """Pipeline should raise ValueError if critical fields are missing."""
        raw = pd.DataFrame([{"build_id": "x", "repo": "r"}])
        with pytest.raises(ValueError, match="timestamp"):
            engineer_features(raw)


# =============================================================================
# 11. Schema validation (data_loader)
# =============================================================================

class TestValidateRawSchema:
    def test_valid_schema(self):
        df = pd.DataFrame([{
            "build_id": "b1", "repo": "r", "branch": "main",
            "commit_sha": "abc", "author": "a",
            "timestamp": "2026-08-01T10:00:00Z",
            "files_changed": 5, "lines_added": 100, "lines_deleted": 20,
            "test_count": 10, "build_duration_seconds": 120,
            "status": "SUCCEEDED", "failed_stage": None,
        }])
        result = validate_raw_schema(df)
        assert result["valid"] is True
        assert result["missing_fields"] == []
        assert result["invalid_statuses"] == []

    def test_missing_field(self):
        df = pd.DataFrame([{"build_id": "b1", "repo": "r", "status": "SUCCEEDED"}])
        result = validate_raw_schema(df)
        assert result["valid"] is False
        assert "timestamp" in result["missing_fields"]

    def test_invalid_status(self):
        df = pd.DataFrame([{
            "build_id": "b1", "repo": "r", "branch": "main",
            "commit_sha": "abc", "author": "a",
            "timestamp": "2026-08-01T10:00:00Z",
            "files_changed": 5, "lines_added": 100, "lines_deleted": 20,
            "test_count": 10, "build_duration_seconds": 120,
            "status": "CANCELLED", "failed_stage": None,
        }])
        result = validate_raw_schema(df)
        assert result["valid"] is False
        assert "CANCELLED" in result["invalid_statuses"]

    def test_null_critical_field(self):
        df = pd.DataFrame([{
            "build_id": None, "repo": "r", "branch": "main",
            "commit_sha": "abc", "author": "a",
            "timestamp": "2026-08-01T10:00:00Z",
            "files_changed": 5, "lines_added": 100, "lines_deleted": 20,
            "test_count": 10, "build_duration_seconds": 120,
            "status": "SUCCEEDED", "failed_stage": None,
        }])
        result = validate_raw_schema(df)
        assert result["valid"] is False
        assert result["null_counts"]["build_id"] == 1

    def test_extra_field_detected(self):
        df = pd.DataFrame([{
            "build_id": "b1", "repo": "r", "branch": "main",
            "commit_sha": "abc", "author": "a",
            "timestamp": "2026-08-01T10:00:00Z",
            "files_changed": 5, "lines_added": 100, "lines_deleted": 20,
            "test_count": 10, "build_duration_seconds": 120,
            "status": "SUCCEEDED", "failed_stage": None,
            "extra_field": "surprise",
        }])
        result = validate_raw_schema(df)
        # Extra fields don't make it invalid, just warned
        assert "extra_field" in result["extra_fields"]


# =============================================================================
# 12. Synthetic data loader test
# =============================================================================

class TestSyntheticDataLoad:
    def test_load_synthetic_data(self):
        """Verify synthetic data loads and passes validation."""
        data_path = _ML_ROOT / "data" / "synthetic" / "raw_builds.json"
        if not data_path.exists():
            pytest.skip("Synthetic data file not found")

        df = load_raw_builds_json(data_path)
        assert len(df) == 30  # 3 repos × 10 builds each

        result = validate_raw_schema(df)
        assert result["valid"] is True
        assert result["missing_fields"] == []
        assert result["invalid_statuses"] == []

    def test_synthetic_data_full_pipeline(self):
        """Run the full pipeline on synthetic data without errors."""
        data_path = _ML_ROOT / "data" / "synthetic" / "raw_builds.json"
        if not data_path.exists():
            pytest.skip("Synthetic data file not found")

        df = load_raw_builds_json(data_path)
        processed, warnings_list = engineer_features(df)

        # Check shape
        assert len(processed) == 30
        assert list(processed.columns) == OUTPUT_COLUMNS

        # Check no unexpected nulls in non-nullable features
        assert processed["time_of_day_bucket"].notna().all()
        assert processed["day_of_week"].notna().all()
        assert processed["author_recent_failure_rate"].notna().all()
        assert processed["repo_recent_failure_rate"].notna().all()
        assert processed["prior_failures_last_5_builds"].notna().all()

        # Check label: should have 0s, 1s, and at least one null
        # pandas stores None as NaN in numeric columns, so use isna()
        labeled_values = processed["label"].dropna().unique().tolist()
        assert 0.0 in labeled_values  # SUCCEEDED
        assert 1.0 in labeled_values  # FAILED
        assert processed["label"].isna().any()  # gamma-010 is IN_PROGRESS

        # Check test_coverage_delta warning was emitted
        coverage_warnings = [w for w in warnings_list if "test_coverage_delta" in w]
        assert len(coverage_warnings) > 0
