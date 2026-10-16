"""Prepare TravisTorrent source columns for Day 2 EDA.

This is a Day 1 inspection step only. It does not calculate project features,
map labels, or train a model.
"""

import argparse
import csv
import gzip
from pathlib import Path


SOURCE_COLUMNS = [
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
]


def prepare_dataset(input_path: Path, output_path: Path) -> tuple[list[str], int]:
    """Copy relevant source columns without engineering project features."""
    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        available = [column for column in SOURCE_COLUMNS if column in (reader.fieldnames or [])]
        if not available:
            raise ValueError("None of the expected TravisTorrent columns were found")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as prepared:
            writer = csv.DictWriter(prepared, fieldnames=available)
            writer.writeheader()
            row_count = 0
            for row in reader:
                writer.writerow({column: row.get(column, "") for column in available})
                row_count += 1

    return available, row_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TravisTorrent for Day 2 EDA")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "travistorrent.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "travistorrent_day1.csv",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}. Place travistorrent.csv.gz in ml/data/."
        )

    columns, rows = prepare_dataset(args.input, args.output)
    print(f"Prepared {rows} rows for Day 2 EDA")
    print(f"Columns retained: {', '.join(columns)}")
    print("No project features were calculated and no model was trained.")


if __name__ == "__main__":
    main()