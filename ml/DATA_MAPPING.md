# TravisTorrent Data Mapping

This is the finalized Day 2 mapping. A feature marked unavailable is omitted
from the model-ready CSV; it is never filled with an invented value.

| Project Feature | TravisTorrent Column(s) | Available? | Transformation | Notes |
|---|---|---|---|
| `files_changed` | `gh_diff_files_added`, `gh_diff_files_deleted`, `gh_diff_files_modified` | Yes | Sum the three file-count columns. | Missing source values remain missing in the sum. |
| `lines_changed` | `git_diff_src_churn`, `git_diff_test_churn` | Yes | Sum source and test churn. | Missing source values remain missing in the sum. |
| `test_coverage_delta` | None | No | Omitted | `gh_test_lines_per_kloc` is test lines per KLOC, not coverage percentage or coverage change. |
| `author_recent_failure_rate` | None | No | Omitted | No stable author identifier was selected or confirmed in the dataset schema. |
| `repo_recent_failure_rate` | `gh_project_name`, `tr_status`, `gh_build_started_at` | Yes | Failed earlier builds / completed earlier builds in the prior 30 days for the same repository. | Current build and all future builds are excluded. Defaults to `0.0` with no history. |
| `time_of_day_bucket` | `gh_build_started_at` | Yes | UTC hour: night 0–5, morning 6–11, afternoon 12–17, evening 18–23. | |
| `day_of_week` | `gh_build_started_at` | Yes | Extract UTC weekday abbreviation. | |
| `prior_failures_last_5_builds` | `gh_project_name`, `tr_status`, `gh_build_started_at` | Yes | Count failures in the five immediately preceding completed builds for the repository. | Current build and future builds are excluded. |
| `label` | `tr_status` | Yes | Normalize passed/success to `0`, failed/error to `1`; drop unrecognized statuses. | Status values are inspected before normalization. |

## Missing-Value Policy

- Rows missing build ID, repository, build timestamp, or a recognized status are
	removed because they cannot be ordered or labeled safely.
- Numeric inputs are parsed with invalid values becoming missing; sums use
	`min_count=1`, so a missing pair does not become a fabricated zero.
- Historical repository rates use `0.0` when no earlier completed builds exist.
- No imputation is performed for unavailable project features.

## Dataset Decision

The real S3 data remains the source for integration. TravisTorrent is a backup
for EDA and later model development because eight real builds are insufficient
for meaningful training. The generated Day 2 dataset is local only; nothing is
uploaded to AWS.