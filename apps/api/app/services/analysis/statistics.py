
"""NUMPA v2 — Dataset statistics engine.

Provides descriptive statistics, distributions, categorical
frequencies and Pearson correlations for dataset analysis.

All results are JSON-compatible and the original DataFrame
is never modified.
"""

from typing import Any

import numpy as np
import pandas as pd

from app.services.analysis.profiler import make_json_safe


# Prevent very large API responses.
DEFAULT_TOP_VALUES = 10
DEFAULT_HISTOGRAM_BINS = 10
MAX_CORRELATION_COLUMNS = 100


def get_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding boolean columns."""

    return [
        str(column)
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
        and not pd.api.types.is_bool_dtype(dataframe[column])
    ]


def calculate_numeric_statistics(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Calculate descriptive statistics for numeric columns."""

    results: dict[str, dict[str, Any]] = {}

    for column in get_numeric_columns(dataframe):
        # Treat infinity as invalid for statistical calculations.
        series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        valid = series.dropna()

        if valid.empty:
            results[column] = {
                "count": 0,
                "missing_count": int(series.isna().sum()),
                "mean": None,
                "median": None,
                "std": None,
                "variance": None,
                "min": None,
                "max": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "skewness": None,
            }
            continue

        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))

        results[column] = {
            "count": int(valid.count()),
            "missing_count": int(series.isna().sum()),
            "mean": make_json_safe(float(valid.mean())),
            "median": make_json_safe(float(valid.median())),
            "std": (
                make_json_safe(float(valid.std()))
                if len(valid) > 1
                else None
            ),
            "variance": (
                make_json_safe(float(valid.var()))
                if len(valid) > 1
                else None
            ),
            "min": make_json_safe(float(valid.min())),
            "max": make_json_safe(float(valid.max())),
            "q1": make_json_safe(q1),
            "q3": make_json_safe(q3),
            "iqr": make_json_safe(q3 - q1),
            "skewness": (
                make_json_safe(float(valid.skew()))
                if len(valid) >= 3
                else None
            ),
        }

    return results


def calculate_histograms(
    dataframe: pd.DataFrame,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict[str, Any]:
    """Generate histogram data for numeric columns."""

    if not 1 <= bins <= 100:
        raise ValueError("bins must be between 1 and 100.")

    results: dict[str, Any] = {}

    for column in get_numeric_columns(dataframe):
        series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        valid = series.dropna()

        if valid.empty:
            results[column] = {
                "bin_edges": [],
                "counts": [],
            }
            continue

        # NumPy returns one more edge than the number of bins.
        counts, edges = np.histogram(
            valid.to_numpy(dtype=float),
            bins=bins,
        )

        results[column] = {
            "bin_edges": [
                make_json_safe(float(edge))
                for edge in edges
            ],
            "counts": [
                int(count)
                for count in counts
            ],
        }

    return results


def calculate_categorical_statistics(
    dataframe: pd.DataFrame,
    top_n: int = DEFAULT_TOP_VALUES,
) -> dict[str, Any]:
    """Calculate frequency distributions for nonnumeric columns."""

    if not 1 <= top_n <= 100:
        raise ValueError("top_n must be between 1 and 100.")

    results: dict[str, Any] = {}
    numeric_columns = set(get_numeric_columns(dataframe))

    for column in dataframe.columns:
        column_name = str(column)

        if column_name in numeric_columns:
            continue

        series = dataframe[column]
        frequencies = series.value_counts(
            dropna=True,
        ).head(top_n)

        results[column_name] = {
            "unique_count": int(series.nunique(dropna=True)),
            "missing_count": int(series.isna().sum()),
            "top_values": [
                {
                    "value": make_json_safe(value),
                    "count": int(count),
                }
                for value, count in frequencies.items()
            ],
        }

    return results


def calculate_correlations(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate pairwise Pearson correlations for numeric columns."""

    numeric_columns = get_numeric_columns(dataframe)

    if len(numeric_columns) > MAX_CORRELATION_COLUMNS:
        return {
            "columns": [],
            "matrix": {},
            "skipped": True,
            "reason": (
                "Too many numeric columns for the initial "
                "correlation analysis."
            ),
        }

    if len(numeric_columns) < 2:
        return {
            "columns": numeric_columns,
            "matrix": {},
            "skipped": True,
            "reason": "At least two numeric columns are required.",
        }

    # Work on a copy to preserve the uploaded dataset.
    numeric_data = dataframe[numeric_columns].copy()

    numeric_data = numeric_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    correlation_matrix = numeric_data.corr(
        method="pearson",
        min_periods=2,
    )

    matrix: dict[str, dict[str, Any]] = {}

    for row_column in correlation_matrix.index:
        matrix[str(row_column)] = {
            str(column): make_json_safe(
                float(correlation_matrix.loc[row_column, column])
            )
            for column in correlation_matrix.columns
        }

    return {
        "columns": numeric_columns,
        "matrix": matrix,
        "skipped": False,
        "reason": None,
    }


def generate_dataset_statistics(
    dataframe: pd.DataFrame,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    top_n: int = DEFAULT_TOP_VALUES,
) -> dict[str, Any]:
    """Generate NUMPA's complete dataset statistics report."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Duplicate column names must be resolved first."
        )

    return {
        "numeric": calculate_numeric_statistics(dataframe),
        "categorical": calculate_categorical_statistics(
            dataframe,
            top_n=top_n,
        ),
        "histograms": calculate_histograms(
            dataframe,
            bins=histogram_bins,
        ),
        "correlations": calculate_correlations(dataframe),
    }
