# %% [markdown]
# # M2 — ML & Prediction Layer: EDA & Feature Pipeline Demo
#
# This notebook demonstrates the M2 feature engineering pipeline:
#
# 1. Load raw synthetic build events (M1 schema, specs.md §4)
# 2. Validate the raw schema
# 3. Run the feature engineering pipeline
# 4. Inspect the processed features (specs.md §5)
# 5. Basic EDA: class distribution, feature ranges, null analysis
#
# **How to run**: Open in VS Code and use "Run Cell" (Ctrl+Shift+Enter),
# or convert to .ipynb with `jupytext --to notebook m2_eda_pipeline.py`.

# %%
import sys
from pathlib import Path

# Ensure ml/ is on the path
_ML_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

import pandas as pd
import numpy as np

from src.data_loader import load_raw_builds_json, validate_raw_schema
from src.feature_engineering import engineer_features
from config.settings import (
    OUTPUT_COLUMNS, FEATURE_COLUMNS,
    TEST_COVERAGE_DATA_AVAILABLE, TEST_COVERAGE_PLACEHOLDER_VALUE,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("M2 EDA Pipeline — imports loaded successfully")

# %% [markdown]
# ## 1. Load Raw Synthetic Data
#
# The synthetic dataset follows the exact raw schema from specs.md §4.
# It contains 30 build events across 3 repositories and 4 authors (+1 null).

# %%
DATA_PATH = _ML_ROOT / "data" / "synthetic" / "raw_builds.json"
raw_df = load_raw_builds_json(DATA_PATH)
print(f"Loaded {len(raw_df)} raw build events\n")
print("--- First 5 rows ---")
print(raw_df.head().to_string())

# %%
print("\n--- Raw schema info ---")
print(raw_df.dtypes)
print(f"\nShape: {raw_df.shape}")

# %% [markdown]
# ## 2. Validate Raw Schema
#
# Check that all fields from specs.md §4 are present and valid.

# %%
validation = validate_raw_schema(raw_df)

print(f"Schema valid: {validation['valid']}")
print(f"Missing fields: {validation['missing_fields']}")
print(f"Extra fields: {validation['extra_fields']}")
print(f"Invalid statuses: {validation['invalid_statuses']}")
print(f"\nNull counts:")
for field, count in validation["null_counts"].items():
    print(f"  {field}: {count} null(s)")

print(f"\nWarnings:")
for w in validation["warnings"]:
    print(f"  • {w}")

# %% [markdown]
# ## 3. Raw Data Summary

# %%
print("--- Status distribution ---")
print(raw_df["status"].value_counts().to_string())

print("\n--- Repos ---")
print(raw_df["repo"].value_counts().to_string())

print("\n--- Authors ---")
print(raw_df["author"].value_counts(dropna=False).to_string())

print("\n--- Timestamp range ---")
timestamps = pd.to_datetime(raw_df["timestamp"])
print(f"  Earliest: {timestamps.min()}")
print(f"  Latest:   {timestamps.max()}")
print(f"  Span:     {(timestamps.max() - timestamps.min()).days} days")

# %% [markdown]
# ## 4. Feature Engineering Pipeline
#
# Transform raw events → processed ML features (specs.md §5).

# %%
processed_df, pipeline_warnings = engineer_features(raw_df)

print(f"Processed {len(processed_df)} records\n")

print("--- Pipeline Warnings ---")
for w in pipeline_warnings:
    print(f"  • {w}")

# %%
print("\n--- Processed features (first 10 rows) ---")
print(processed_df.head(10).to_string())

# %%
print("\n--- Column check ---")
print(f"Expected columns: {OUTPUT_COLUMNS}")
print(f"Actual columns:   {list(processed_df.columns)}")
print(f"Match: {list(processed_df.columns) == OUTPUT_COLUMNS}")

# %% [markdown]
# ## 5. Class Distribution
#
# Label mapping (specs.md §5 L102-106):
# - 0 = SUCCEEDED
# - 1 = FAILED
# - None = IN_PROGRESS (unlabeled)

# %%
print("--- Label distribution ---")
label_counts = processed_df["label"].value_counts(dropna=False)
print(label_counts.to_string())

total_labeled = processed_df["label"].notna().sum()
if total_labeled > 0:
    fail_rate = (processed_df["label"] == 1).sum() / total_labeled
    print(f"\nFailure rate (labeled builds): {fail_rate:.1%}")
    print(f"Unlabeled (IN_PROGRESS): {processed_df['label'].isna().sum()}")

# %% [markdown]
# ## 6. Feature Inspection

# %%
print("--- Numeric feature statistics ---")
numeric_features = [
    "files_changed", "lines_changed", "test_coverage_delta",
    "author_recent_failure_rate", "repo_recent_failure_rate",
    "prior_failures_last_5_builds",
]
print(processed_df[numeric_features].describe().to_string())

# %%
print("\n--- test_coverage_delta status ---")
if TEST_COVERAGE_DATA_AVAILABLE:
    print("  ✅ Real coverage data is available.")
else:
    print(f"  ⚠️  PLACEHOLDER: All values set to {TEST_COVERAGE_PLACEHOLDER_VALUE}")
    print("  Raw schema (specs.md §4) has 'test_count' but NOT coverage percentage.")
    print("  DEPENDENCY: M1 must add coverage data for this feature to be real.")

# %%
print("\n--- Null counts in processed features ---")
null_counts = processed_df[FEATURE_COLUMNS].isna().sum()
for col, count in null_counts.items():
    status = f"⚠️  {count} null(s)" if count > 0 else "✅ no nulls"
    print(f"  {col}: {status}")

# %%
print("\n--- time_of_day_bucket distribution ---")
print(processed_df["time_of_day_bucket"].value_counts().to_string())

print("\n--- day_of_week distribution ---")
print(processed_df["day_of_week"].value_counts().to_string())

# %%
print("\n--- prior_failures_last_5_builds distribution ---")
print(processed_df["prior_failures_last_5_builds"].value_counts().sort_index().to_string())

# %%
print("\n--- Failure rate features (non-zero values) ---")
rate_cols = ["author_recent_failure_rate", "repo_recent_failure_rate"]
for col in rate_cols:
    non_zero = processed_df[processed_df[col] > 0][["build_id", "repo", col]]
    if len(non_zero) > 0:
        print(f"\n{col} > 0:")
        print(non_zero.to_string())
    else:
        print(f"\n{col}: all zeros (no prior history or no failures)")

# %% [markdown]
# ## 7. Per-Repo Summary

# %%
print("--- Per-repo build summary ---")
for repo in processed_df["repo"].unique():
    repo_data = processed_df[processed_df["repo"] == repo]
    labeled = repo_data["label"].dropna()
    fail_count = (labeled == 1).sum()
    total = len(labeled)
    print(f"\n  {repo}:")
    print(f"    Total builds: {len(repo_data)}")
    print(f"    Labeled: {total}, Failed: {fail_count}, "
          f"Rate: {fail_count/total:.1%}" if total > 0 else "    No labeled builds")

# %% [markdown]
# ## 8. Data Quality Summary
#
# This section summarizes what M2 can and cannot compute from current data.

# %%
print("=" * 60)
print("M2 DATA QUALITY SUMMARY")
print("=" * 60)
print()
print("Features fully computable from raw data:")
print("  ✅ files_changed (direct passthrough)")
print("  ✅ lines_changed (lines_added + lines_deleted)")
print("  ✅ author_recent_failure_rate (trailing 30 days)")
print("  ✅ repo_recent_failure_rate (trailing 30 days)")
print("  ✅ time_of_day_bucket (from timestamp)")
print("  ✅ day_of_week (from timestamp)")
print("  ✅ prior_failures_last_5_builds (from history)")
print("  ✅ label (from status)")
print()
print("Features with known limitations:")
print(f"  ⚠️  test_coverage_delta: PLACEHOLDER ({TEST_COVERAGE_PLACEHOLDER_VALUE})")
print("      → Raw schema has 'test_count', not coverage percentage")
print("      → DEPENDENCY: M1 must add coverage data")
print()
print("Data quality notes:")
null_files = processed_df["files_changed"].isna().sum()
null_lines = processed_df["lines_changed"].isna().sum()
if null_files > 0 or null_lines > 0:
    print(f"  ⚠️  {null_files} records have null files_changed")
    print(f"  ⚠️  {null_lines} records have null lines_changed")
    print("      → These are real nulls from M1, not fabricated values")
print()
print("Pipeline completed successfully ✅")
