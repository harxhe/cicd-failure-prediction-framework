# TravisTorrent Day 2 EDA Report

## Dataset

- Records read: **3,881,992**
- Unique repositories: **948**
- Build start range: **2011-04-16 11:24:39+00:00** to **2016-08-31 23:59:27+00:00**
- Rows retained for training features: **3,841,797**
- Rows removed: **40,195** (missing build ID, repository, timestamp, or recognized label)

## Labels

Raw normalized label counts: `{0.0: 2810783, 1.0: 1031014, nan: 40195}`. The output contains:

```text
label
0    2810783
1    1031014
```

## Missing Values

Missing counts and percentages for selected source columns:

|                          |          count |   percent |
|:-------------------------|---------------:|----------:|
| tr_build_id              |    0           |     0     |
| gh_project_name          |    0           |     0     |
| git_diff_src_churn       |    0           |     0     |
| git_diff_test_churn      |    0           |     0     |
| gh_diff_files_added      |    0           |     0     |
| gh_diff_files_deleted    |    0           |     0     |
| gh_diff_files_modified   |    0           |     0     |
| gh_test_lines_per_kloc   |    0           |     0     |
| gh_pushed_at             |    3.88199e+06 |   100     |
| gh_build_started_at      |    0           |     0     |
| tr_status                |    0           |     0     |
| tr_log_num_tests_ok      |    1.55686e+06 |    40.105 |
| tr_log_num_tests_failed  |    1.4943e+06  |    38.493 |
| tr_log_num_tests_run     |    1.67542e+06 |    43.159 |
| tr_log_num_tests_skipped |    1.55295e+06 |    40.004 |
| tr_log_buildduration     |    3.69999e+06 |    95.312 |
| tr_duration              | 3209           |     0.083 |

Rows with missing required identity/timestamp/status values are removed. Numeric
source values remain missing and are represented as missing in derived sums;
historical repository rates default to `0.0` when no earlier completed builds
exist. No values are fabricated for unavailable coverage or author identity.

## Numerical Statistics

|                          |            count |     mean |       std |   min |     25% |      50% |     75% |              max |
|:-------------------------|-----------------:|---------:|----------:|------:|--------:|---------:|--------:|-----------------:|
| git_diff_src_churn       |      3.88199e+06 |    1.113 |    23.596 |     0 |   0     |    0     |    0    |  13870           |
| git_diff_test_churn      |      3.88199e+06 |    0     |     0     |     0 |   0     |    0     |    0    |      0           |
| gh_diff_files_added      |      3.88199e+06 |    0.836 |    17.451 |     0 |   0     |    0     |    0    |  19892           |
| gh_diff_files_deleted    |      3.88199e+06 |    0.365 |     9.337 |     0 |   0     |    0     |    0    |   8450           |
| gh_diff_files_modified   |      3.88199e+06 |    4.661 |    22.071 |     0 |   1     |    2     |    3    |   9233           |
| gh_test_lines_per_kloc   |      3.88199e+06 | 1029.24  |  1561.48  |     0 | 296.803 |  682.063 | 1363.58 | 414077           |
| tr_log_num_tests_ok      |      2.32513e+06 | 1797.44  | 17224.7   |     0 |  54     |  330     | 1120    | 733960           |
| tr_log_num_tests_failed  |      2.38769e+06 |    5.672 |    50.152 |     0 |   0     |    0     |    0    |   2800           |
| tr_log_num_tests_run     |      2.20657e+06 | 1900.02  | 17676.1   |     0 |  86     |  373     | 1204    | 733960           |
| tr_log_num_tests_skipped |      2.32904e+06 |  193.163 |  3152.78  |     0 |   0     |    0     |    3    |  59470           |
| tr_log_buildduration     | 182006           |  315.943 |   534.368 |     0 |  44     |  119     |  295    |   3598           |
| tr_duration              |      3.87878e+06 | 4627.27  | 10411.8   |     0 | 581     | 1720     | 5196    |      3.62183e+06 |

## Observations and Limitations

- Source churn, test churn, changed-file counts, test-log fields, and duration
  were inspected from the relevant columns only.
- The selected derived features have useful variation: `files_changed` ranges
  from 0 to 28,591, `lines_changed` ranges from 0 to 13,870,
  `repo_recent_failure_rate` ranges from 0.0 to 1.0, and
  `prior_failures_last_5_builds` ranges from 0 to 5.
- `git_diff_test_churn` is constant at 0 in this dataset, so it contributes no
  independent variation; it is retained only as an input to the documented
  `lines_changed` transformation.
- The output label has both classes: 2,810,783 successful builds and 1,031,014
  failed builds.
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
