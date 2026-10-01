
"""NUMPA v2 - Dataset analysis consistency tests.

Tests:
- Dataset profiling preserves the original data.
- Validation identifies missing values.
- Validation identifies duplicate records.
- Statistics generation preserves the original data.
- Numeric statistics reflect the source dataset.
- Missing-value counts remain consistent.
- Empty datasets are handled safely.
- Repeated analysis produces deterministic results.

These tests exercise the existing analysis services.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.services.analysis.profiler import (
    generate_dataset_profile,
)
from app.services.analysis.statistics import (
    generate_dataset_statistics,
)
from app.services.analysis.validator import (
    validate_dataset,
)


# ---------------------------------------------------------
# Test data
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Create a deterministic dataset with common data issues."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 103, 105],
            "age": [24.0, None, 38.0, 38.0, 45.0],
            "income": [
                25000.0,
                42000.0,
                51000.0,
                51000.0,
                63000.0,
            ],
            "city": [
                "Surat",
                "Rajkot",
                "Ahmedabad",
                "Ahmedabad",
                "Vadodara",
            ],
            "churn": [0, 1, 0, 0, 1],
        }
    )


def serialized(value):
    """Convert Pydantic models to JSON-compatible dictionaries."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    return value


def find_named_column(
    data,
    column_name: str,
):
    """Find a column profile in common profile containers."""

    data = serialized(data)

    if isinstance(data, dict):
        # Handle dictionaries indexed by column name.
        if column_name in data:
            return serialized(data[column_name])

        # Handle lists stored under common container keys.
        for key in (
            "columns",
            "column_profiles",
            "numeric_columns",
            "categorical_columns",
            "numeric",
            "categorical",
            "statistics",
        ):
            if key in data:
                result = find_named_column(
                    data[key],
                    column_name,
                )

                if result is not None:
                    return result

    if isinstance(data, list):
        for item in data:
            item = serialized(item)

            if not isinstance(item, dict):
                continue

            if (
                item.get("name") == column_name
                or item.get("column") == column_name
                or item.get("column_name") == column_name
            ):
                return item

    return None


# ---------------------------------------------------------
# Test 1: Profile preserves input
# ---------------------------------------------------------

def test_profile_does_not_mutate_dataframe() -> None:
    """Generating a profile must not change the source data."""

    dataframe = sample_dataframe()
    snapshot = dataframe.copy(deep=True)

    profile = generate_dataset_profile(
        dataframe,
        preview_rows=3,
    )

    assert profile is not None

    assert_frame_equal(
        dataframe,
        snapshot,
    )


# ---------------------------------------------------------
# Test 2: Validation detects missing values
# ---------------------------------------------------------

def test_validation_detects_missing_values() -> None:
    """The validator must flag a dataset containing null values."""

    result = validate_dataset(
        sample_dataframe()
    )

    assert isinstance(result, dict)
    assert result["rows"] == 5
    assert result["columns"] == 5

    assert result["issue_count"] > 0

    # Inspect the structured issues without assuming
    # one particular issue-code naming convention.
    issue_text = str(result["issues"]).lower()

    assert (
        "missing" in issue_text
        or "null" in issue_text
    )


# ---------------------------------------------------------
# Test 3: Validation detects duplicate records
# ---------------------------------------------------------

def test_validation_detects_duplicate_records() -> None:
    """Duplicate rows must be visible in validation findings."""

    result = validate_dataset(
        sample_dataframe()
    )

    issue_text = str(result["issues"]).lower()

    assert "duplicate" in issue_text


# ---------------------------------------------------------
# Test 4: Statistics preserve input
# ---------------------------------------------------------

def test_statistics_do_not_mutate_dataframe() -> None:
    """Statistics generation must not modify the source data."""

    dataframe = sample_dataframe()
    snapshot = dataframe.copy(deep=True)

    statistics = generate_dataset_statistics(
        dataframe
    )

    assert isinstance(statistics, dict)

    assert_frame_equal(
        dataframe,
        snapshot,
    )


# ---------------------------------------------------------
# Test 5: Numeric statistics
# ---------------------------------------------------------

def test_numeric_statistics_include_age() -> None:
    """The statistics output must describe numeric columns."""

    dataframe = sample_dataframe()

    statistics = generate_dataset_statistics(
        dataframe
    )

    age_statistics = find_named_column(
        statistics,
        "age",
    )

    assert age_statistics is not None

    # Check numeric facts against pandas rather
    # than hardcoding an expected mean.
    expected_mean = dataframe["age"].mean()

    if "mean" in age_statistics:
        assert age_statistics["mean"] == pytest.approx(
            expected_mean
        )


# ---------------------------------------------------------
# Test 6: Missing values remain consistent
# ---------------------------------------------------------

def test_profile_missing_count_matches_source() -> None:
    """The profile must report the dataset's missing values."""

    dataframe = sample_dataframe()

    profile = generate_dataset_profile(
        dataframe,
        preview_rows=3,
    )

    age_profile = find_named_column(
        profile,
        "age",
    )

    assert age_profile is not None

    expected_missing = int(
        dataframe["age"].isna().sum()
    )

    # The profiler may expose either field name.
    if "missing_count" in age_profile:
        actual_missing = age_profile["missing_count"]

    elif "null_count" in age_profile:
        actual_missing = age_profile["null_count"]

    else:
        pytest.fail(
            "Age profile does not expose a missing-value count."
        )

    assert actual_missing == expected_missing


# ---------------------------------------------------------
# Test 7: Empty dataset
# ---------------------------------------------------------

def test_empty_dataset_validation_is_safe() -> None:
    """Empty datasets must produce structured validation output."""

    empty = pd.DataFrame(
        {
            "age": pd.Series(dtype="float64"),
            "city": pd.Series(dtype="object"),
        }
    )

    result = validate_dataset(empty)

    assert isinstance(result, dict)
    assert result["rows"] == 0
    assert result["valid"] is False
    assert result["issue_count"] > 0


# ---------------------------------------------------------
# Test 8: Deterministic analysis
# ---------------------------------------------------------

def test_repeated_analysis_is_deterministic() -> None:
    """Identical input must produce identical analysis results."""

    dataframe = sample_dataframe()

    first_validation = validate_dataset(
        dataframe.copy(deep=True)
    )

    second_validation = validate_dataset(
        dataframe.copy(deep=True)
    )

    first_statistics = generate_dataset_statistics(
        dataframe.copy(deep=True)
    )

    second_statistics = generate_dataset_statistics(
        dataframe.copy(deep=True)
    )

    assert first_validation == second_validation
    assert first_statistics == second_statistics
