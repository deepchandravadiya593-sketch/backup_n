
"""NUMPA v2 - Cleaning pipeline determinism and data integrity tests.

Coverage:
1. An empty pipeline preserves the dataset.
2. An empty pipeline does not mutate its input.
3. Duplicate removal produces deterministic output.
4. Duplicate removal preserves the original dataframe.
5. Duplicate removal keeps the expected records.
6. Repeated pipeline executions are independent.
7. Preview does not mutate the original dataset.
8. An empty pipeline preserves column order and data types.

The tests use the existing cleaning pipeline service directly.
"""

from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


# ---------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Create a dataset containing two duplicate records."""

    return pd.DataFrame(
        {
            "customer_id": [
                101, 102, 103, 103, 104, 104,
            ],
            "age": [
                24, 31, 38, 38, 45, 45,
            ],
            "city": [
                "Surat",
                "Rajkot",
                "Ahmedabad",
                "Ahmedabad",
                "Vadodara",
                "Vadodara",
            ],
            "spend": [
                1200.0,
                1800.0,
                2400.0,
                2400.0,
                3100.0,
                3100.0,
            ],
        }
    )


def empty_pipeline() -> CleaningPipelineConfig:
    """Create a pipeline without cleaning operations."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Empty Integrity Pipeline",
            "steps": [],
        }
    )


def duplicate_pipeline() -> CleaningPipelineConfig:
    """Create a pipeline that removes duplicate records."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Duplicate Integrity Pipeline",
            "steps": [
                {
                    "step_id": "remove-duplicates",
                    "operation": "duplicates",
                    "parameters": {
                        "strategy": "keep_first",
                    },
                }
            ],
        }
    )


def run_pipeline(
    dataframe: pd.DataFrame,
    pipeline: CleaningPipelineConfig,
    *,
    preview: bool = False,
) -> pd.DataFrame:
    """Execute a pipeline and return its cleaned dataframe."""

    cleaned, _report, _artifacts = execute_pipeline(
        dataframe,
        pipeline,
        preview=preview,
    )

    return cleaned


# ---------------------------------------------------------
# Test 1: Empty pipeline preserves the dataset
# ---------------------------------------------------------

def test_empty_pipeline_preserves_dataset() -> None:
    """No cleaning operations should leave the data unchanged."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        empty_pipeline(),
    )

    assert_frame_equal(
        cleaned.reset_index(drop=True),
        dataframe.reset_index(drop=True),
    )


# ---------------------------------------------------------
# Test 2: Empty pipeline does not mutate its input
# ---------------------------------------------------------

def test_empty_pipeline_does_not_mutate_input() -> None:
    """Executing an empty pipeline must preserve the source."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_pipeline(
        dataframe,
        empty_pipeline(),
    )

    assert_frame_equal(
        dataframe,
        original,
    )


# ---------------------------------------------------------
# Test 3: Duplicate removal is deterministic
# ---------------------------------------------------------

def test_duplicate_removal_is_deterministic() -> None:
    """Identical inputs must produce identical cleaned datasets."""

    dataframe = sample_dataframe()
    pipeline = duplicate_pipeline()

    first = run_pipeline(
        dataframe.copy(deep=True),
        pipeline,
    )

    second = run_pipeline(
        dataframe.copy(deep=True),
        pipeline,
    )

    assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
    )


# ---------------------------------------------------------
# Test 4: Duplicate removal preserves the source
# ---------------------------------------------------------

def test_duplicate_removal_does_not_mutate_input() -> None:
    """Removing duplicates must not alter the uploaded dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_pipeline(
        dataframe,
        duplicate_pipeline(),
    )

    assert_frame_equal(
        dataframe,
        original,
    )


# ---------------------------------------------------------
# Test 5: Duplicate removal keeps expected records
# ---------------------------------------------------------

def test_duplicate_removal_keeps_expected_records() -> None:
    """The first occurrence of each duplicate must be retained."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        duplicate_pipeline(),
    )

    expected = dataframe.drop_duplicates(
        keep="first",
    )

    assert_frame_equal(
        cleaned.reset_index(drop=True),
        expected.reset_index(drop=True),
    )

    assert len(cleaned) == 4


# ---------------------------------------------------------
# Test 6: Pipeline executions are independent
# ---------------------------------------------------------

def test_pipeline_executions_are_independent() -> None:
    """Modifying one result must not affect another execution."""

    dataframe = sample_dataframe()
    pipeline = duplicate_pipeline()

    first = run_pipeline(
        dataframe.copy(deep=True),
        pipeline,
    )

    second = run_pipeline(
        dataframe.copy(deep=True),
        pipeline,
    )

    # Change the first result after both executions.
    first.iloc[0, first.columns.get_loc("age")] = 999

    # The second result and source must remain unchanged.
    assert second.iloc[0]["age"] == 24
    assert dataframe.iloc[0]["age"] == 24


# ---------------------------------------------------------
# Test 7: Preview preserves the original dataset
# ---------------------------------------------------------

def test_preview_does_not_mutate_input() -> None:
    """A cleaning preview must not modify the uploaded data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_pipeline(
        dataframe,
        duplicate_pipeline(),
        preview=True,
    )

    assert_frame_equal(
        dataframe,
        original,
    )


# ---------------------------------------------------------
# Test 8: Empty pipeline preserves the schema
# ---------------------------------------------------------

def test_empty_pipeline_preserves_column_schema() -> None:
    """An empty pipeline must retain column order and data types."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        empty_pipeline(),
    )

    assert list(cleaned.columns) == list(dataframe.columns)

    assert cleaned.dtypes.equals(dataframe.dtypes)

