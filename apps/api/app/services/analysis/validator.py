
"""NUMPA v2 — Dataset quality validator.

Detects common data quality issues before cleaning or ML assessment.
Validation never modifies the original dataset.
"""

from typing import Any

import numpy as np
import pandas as pd


def validate_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a structured report of dataset quality issues."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    rows, columns = dataframe.shape
    issues: list[dict[str, Any]] = []

    # Check whether the dataset contains usable data.
    if rows == 0 or columns == 0:
        issues.append({
            "code": "EMPTY_DATASET",
            "severity": "critical",
            "column": None,
            "message": "The dataset contains no usable rows or columns.",
            "affected_rows": rows,
        })

        return {
            "valid": False,
            "rows": rows,
            "columns": columns,
            "issue_count": len(issues),
            "issues": issues,
        }

    # Detect duplicate column names.
    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    for column in dict.fromkeys(duplicate_columns):
        issues.append({
            "code": "DUPLICATE_COLUMN",
            "severity": "critical",
            "column": str(column),
            "message": f"Duplicate column name: {column}",
            "affected_rows": rows,
        })

    # Detect missing values and problematic columns.
    for index in range(columns):
        series = dataframe.iloc[:, index]
        column_name = str(dataframe.columns[index])

        missing_count = int(series.isna().sum())

        if missing_count == rows:
            issues.append({
                "code": "EMPTY_COLUMN",
                "severity": "high",
                "column": column_name,
                "message": "Every value in this column is missing.",
                "affected_rows": rows,
            })

        elif missing_count > 0:
            issues.append({
                "code": "MISSING_VALUES",
                "severity": "medium",
                "column": column_name,
                "message": (
                    f"{missing_count} missing values detected."
                ),
                "affected_rows": missing_count,
            })

        # A constant column contains only one distinct
        # non-null value and may be uninformative for ML.
        non_null = series.dropna()

        if (
            not non_null.empty
            and non_null.nunique(dropna=True) == 1
        ):
            issues.append({
                "code": "CONSTANT_COLUMN",
                "severity": "low",
                "column": column_name,
                "message": (
                    "This column contains only one distinct "
                    "non-null value."
                ),
                "affected_rows": int(non_null.shape[0]),
            })

        # Detect positive and negative infinity in numeric columns.
        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            infinity_count = int(
                np.isinf(numeric.to_numpy(dtype=float)).sum()
            )

            if infinity_count > 0:
                issues.append({
                    "code": "INFINITE_VALUES",
                    "severity": "high",
                    "column": column_name,
                    "message": (
                        f"{infinity_count} infinite numeric "
                        "values detected."
                    ),
                    "affected_rows": infinity_count,
                })

    # Duplicate-row checks are unreliable when column
    # names are duplicated or cells contain unhashable objects.
    if not duplicate_columns:
        try:
            duplicate_count = int(
                dataframe.duplicated().sum()
            )

            if duplicate_count > 0:
                issues.append({
                    "code": "DUPLICATE_ROWS",
                    "severity": "medium",
                    "column": None,
                    "message": (
                        f"{duplicate_count} duplicate rows detected."
                    ),
                    "affected_rows": duplicate_count,
                })

        except TypeError:
            issues.append({
                "code": "DUPLICATE_CHECK_SKIPPED",
                "severity": "info",
                "column": None,
                "message": (
                    "Duplicate-row detection could not be "
                    "completed for this dataset."
                ),
                "affected_rows": 0,
            })

    # Critical issues prevent further processing.
    has_critical_issues = any(
        issue["severity"] == "critical"
        for issue in issues
    )

    return {
        "valid": not has_critical_issues,
        "rows": rows,
        "columns": columns,
        "issue_count": len(issues),
        "issues": issues,
    }
