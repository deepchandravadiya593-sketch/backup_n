
"""NUMPA v2 — Dataset profiling engine.

Generates a JSON-compatible overview of an uploaded dataset.
All calculations are deterministic and do not modify the dataset.
"""

from typing import Any

import numpy as np
import pandas as pd


# Limit the number of preview rows returned to the frontend.
DEFAULT_PREVIEW_ROWS = 10


def make_json_safe(value: Any) -> Any:
    """Convert pandas and NumPy values into JSON-compatible values."""

    # Convert missing values into JSON null.
    if value is None:
        return None

    if isinstance(value, (dict, list, tuple)):
        if isinstance(value, dict):
            return {
                str(key): make_json_safe(item)
                for key, item in value.items()
            }

        return [make_json_safe(item) for item in value]

    # Convert NumPy scalar values into native Python values.
    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    # Convert dates and timestamps into ISO-formatted strings.
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    # Convert non-finite numbers and missing scalar values.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def detect_column_type(series: pd.Series) -> str:
    """Classify a column into a useful NUMPA data category."""

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    # String columns with relatively few unique values are
    # potential categorical features.
    non_null = series.dropna()

    if non_null.empty:
        return "unknown"

    unique_count = non_null.nunique()

    if unique_count <= 20:
        return "categorical"

    return "text"


def profile_column(
    series: pd.Series,
    total_rows: int,
) -> dict[str, Any]:
    """Calculate quality and statistical information for one column."""

    missing_count = int(series.isna().sum())
    non_null_count = int(series.notna().sum())

    # Avoid division by zero for empty datasets.
    missing_percentage = (
        round(missing_count / total_rows * 100, 2)
        if total_rows > 0
        else 0.0
    )

    # Include nulls when counting distinct values.
    unique_count = int(series.nunique(dropna=True))

    column_profile: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "detected_type": detect_column_type(series),
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "non_null_count": non_null_count,
        "unique_count": unique_count,
    }

    # Calculate numeric statistics only for numeric columns.
    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    ):
        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        valid = numeric_series.dropna()

        column_profile["statistics"] = {
            "mean": (
                make_json_safe(float(valid.mean()))
                if not valid.empty
                else None
            ),
            "median": (
                make_json_safe(float(valid.median()))
                if not valid.empty
                else None
            ),
            "std": (
                make_json_safe(float(valid.std()))
                if len(valid) > 1
                else None
            ),
            "min": (
                make_json_safe(float(valid.min()))
                if not valid.empty
                else None
            ),
            "max": (
                make_json_safe(float(valid.max()))
                if not valid.empty
                else None
            ),
        }

    # Return the most common values for categorical columns.
    elif column_profile["detected_type"] == "categorical":
        frequencies = series.value_counts(
            dropna=True,
        ).head(5)

        column_profile["top_values"] = [
            {
                "value": make_json_safe(value),
                "count": int(count),
            }
            for value, count in frequencies.items()
        ]

    return column_profile


def generate_dataset_profile(
    dataframe: pd.DataFrame,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
) -> dict[str, Any]:
    """Generate the complete NUMPA dataset overview."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot profile an empty dataset.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Cannot profile a dataset with duplicate column names."
        )

    if preview_rows < 0:
        raise ValueError("preview_rows cannot be negative.")

    total_rows, total_columns = dataframe.shape

    # Calculate dataset-wide quality metrics.
    total_cells = total_rows * total_columns
    missing_cells = int(dataframe.isna().sum().sum())

    missing_percentage = (
        round(missing_cells / total_cells * 100, 2)
        if total_cells > 0
        else 0.0
    )

    duplicate_rows = int(dataframe.duplicated().sum())

    duplicate_percentage = (
        round(duplicate_rows / total_rows * 100, 2)
        if total_rows > 0
        else 0.0
    )

    # Generate a profile for every column.
    columns = [
        profile_column(dataframe[column], total_rows)
        for column in dataframe.columns
    ]

    # Convert preview records into JSON-safe values.
    preview = [
        {
            str(key): make_json_safe(value)
            for key, value in record.items()
        }
        for record in dataframe.head(preview_rows).to_dict(
            orient="records"
        )
    ]

    return {
        "overview": {
            "rows": total_rows,
            "columns": total_columns,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": missing_percentage,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "memory_usage_bytes": int(
                dataframe.memory_usage(deep=True).sum()
            ),
        },
        "columns": columns,
        "preview": preview,
    }
