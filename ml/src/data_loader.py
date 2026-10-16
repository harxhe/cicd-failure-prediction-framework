"""
M2 Data Loader Module

Loads raw CI/CD build events from local files (JSON array, newline-delimited
JSON). Designed so the loading interface can be extended for S3 sources later
without changing downstream code.

Raw event schema: specs.md §4 (13 locked fields).
"""

import json
import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow imports from ml/ root regardless of invocation method
# ---------------------------------------------------------------------------
import sys
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from config.settings import RAW_SCHEMA_FIELDS, VALID_STATUSES


# =============================================================================
# Loaders
# =============================================================================

def load_raw_builds_json(filepath):
    """
    Load raw build events from a JSON file containing an array of objects.

    Args:
        filepath: Path to JSON file.

    Returns:
        pandas DataFrame with raw build events.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file does not contain a JSON array.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected JSON array of build events, got {type(data).__name__}"
        )

    if len(data) == 0:
        warnings.warn("Empty dataset loaded — 0 records")
        return pd.DataFrame(columns=RAW_SCHEMA_FIELDS)

    return pd.DataFrame(data)


def load_raw_builds_ndjson(filepath):
    """
    Load raw build events from newline-delimited JSON (one object per line).

    Args:
        filepath: Path to NDJSON file.

    Returns:
        pandas DataFrame with raw build events.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                warnings.warn(f"Skipping malformed JSON at line {line_num}: {e}")

    if len(records) == 0:
        warnings.warn("Empty dataset loaded — 0 records")
        return pd.DataFrame(columns=RAW_SCHEMA_FIELDS)

    return pd.DataFrame(records)


# =============================================================================
# Validation
# =============================================================================

def validate_raw_schema(df):
    """
    Validate a DataFrame against the raw build event schema (specs.md §4).

    Checks:
    - Required fields present
    - Extra/unexpected fields
    - Null counts per field
    - Critical fields (build_id, repo, timestamp, status) not null
    - Status values in allowed enum

    Args:
        df: pandas DataFrame to validate.

    Returns:
        dict with keys:
            valid (bool): True if no critical violations.
            missing_fields (list): Fields from specs.md §4 not in the DataFrame.
            extra_fields (list): Fields in the DataFrame not in specs.md §4.
            null_counts (dict): {field: count} for fields with nulls.
            invalid_statuses (list): Status values not in VALID_STATUSES.
            warnings (list): Human-readable warning strings.
    """
    result = {
        "valid": True,
        "missing_fields": [],
        "extra_fields": [],
        "null_counts": {},
        "invalid_statuses": [],
        "warnings": [],
    }

    # --- Missing required fields ---
    for field in RAW_SCHEMA_FIELDS:
        if field not in df.columns:
            result["missing_fields"].append(field)
            result["valid"] = False
            result["warnings"].append(
                f"MISSING: Required field '{field}' (specs.md §4) not found."
            )

    # --- Extra fields (informational) ---
    for field in df.columns:
        if field not in RAW_SCHEMA_FIELDS:
            result["extra_fields"].append(field)
            result["warnings"].append(
                f"EXTRA: Field '{field}' not in specs.md §4 raw schema."
            )

    # --- Null counts ---
    critical_fields = {"build_id", "repo", "timestamp", "status"}
    for field in RAW_SCHEMA_FIELDS:
        if field in df.columns:
            null_count = int(df[field].isna().sum())
            if null_count > 0:
                result["null_counts"][field] = null_count
                if field in critical_fields:
                    result["valid"] = False
                    result["warnings"].append(
                        f"CRITICAL: '{field}' has {null_count} null value(s) — "
                        f"this field must not be null."
                    )
                else:
                    result["warnings"].append(
                        f"INFO: '{field}' has {null_count} null value(s) "
                        f"out of {len(df)} records."
                    )

    # --- Status enum ---
    if "status" in df.columns:
        statuses = df["status"].dropna().unique().tolist()
        for s in statuses:
            if s not in VALID_STATUSES:
                result["invalid_statuses"].append(s)
                result["valid"] = False
                result["warnings"].append(
                    f"INVALID: Status '{s}' not in allowed values "
                    f"{VALID_STATUSES} (specs.md §4 L77)."
                )

    return result
