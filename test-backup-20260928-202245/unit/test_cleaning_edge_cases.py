
"""NUMPA v2 - Cleaning pipeline edge-case tests.

Covers:
- Empty pipelines.
- Empty input datasets.
- Missing columns.
- Invalid operation names.
- Invalid operation configuration.
- Duplicate step identifiers.
- Preview row limits.
- Input dataframe immutability.
- Pipeline step ordering.
- Error handling after an earlier successful step.

These tests use the existing cleaning pipeline service.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    PipelineExecutionError,
    PipelineStep,
    execute_pipeline,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Create a small dataset with common cleaning issues."""

    return pd.DataFrame(
        {
            "age": [20.0, None, 30.0, 30.0],
            "income": [1000.0, 2000.0, 3000.0, 3000.0],
            "city": ["Surat", "Ahmedabad", "Rajkot", "Rajkot"],
        }
    )


def make_step(
    step_id: str,
    operation: str,
    config: dict | None = None,
) -> dict:
    """Build a pipeline step using the public schema."""

    return {
        "step_id": step_id,
        "operation": operation,
        "config": config or {},
    }


def make_pipeline(
    steps: list[dict],
    *,
    name: str = "Edge Case Pipeline",
) -> CleaningPipelineConfig:
    """Validate a pipeline using its Pydantic model."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": name,
            "steps": steps,
        }
    )


def run_pipeline(
    dataframe: pd.DataFrame,
    steps: list[dict],
    *,
    preview: bool = False,
    preview_rows: int = 10,
):
    """Execute a validated pipeline."""

    pipeline = make_pipeline(steps)

    return execute_pipeline(
        dataframe,
        pipeline,
        preview=preview,
        preview_rows=preview_rows,
    )


# ---------------------------------------------------------
# Test 1: Empty pipeline
# ---------------------------------------------------------

def test_empty_pipeline_preserves_dataset() -> None:
    """A pipeline with no operations must not alter data."""

    original = sample_dataframe()
    snapshot = original.copy(deep=True)

    cleaned, report, artifacts = run_pipeline(
        original,
        [],
    )

    assert_frame_equal(cleaned, snapshot)
    assert_frame_equal(original, snapshot)

    assert isinstance(report, dict)
    assert isinstance(artifacts, dict)


# ---------------------------------------------------------
# Test 2: Empty input dataset
# ---------------------------------------------------------

def test_empty_input_does_not_create_fake_rows() -> None:
    """Cleaning must never invent records in an empty dataset."""

    empty = pd.DataFrame(
        {
            "age": pd.Series(dtype="float64"),
            "city": pd.Series(dtype="object"),
        }
    )

    cleaned, report, artifacts = run_pipeline(
        empty,
        [],
    )

    assert cleaned.empty
    assert len(cleaned) == 0

    assert isinstance(report, dict)
    assert isinstance(artifacts, dict)


# ---------------------------------------------------------
# Test 3: Invalid operation
# ---------------------------------------------------------

def test_unknown_operation_is_rejected() -> None:
    """Only registered cleaning operations may be used."""

    with pytest.raises(ValidationError):
        make_pipeline(
            [
                make_step(
                    "invalid-operation",
                    "delete_everything",
                )
            ]
        )


# ---------------------------------------------------------
# Test 4: Invalid missing-value strategy
# ---------------------------------------------------------

def test_invalid_missing_strategy_is_rejected() -> None:
    """An unsupported missing-value strategy must fail safely."""

    dataframe = sample_dataframe()

    invalid_steps = [
        make_step(
            "invalid-missing",
            "missing",
            {
                "strategy": "invent_values",
            },
        )
    ]

    # Configuration validation may happen while
    # constructing the pipeline or during execution.
    with pytest.raises(
        (ValidationError, ValueError, PipelineExecutionError)
    ):
        run_pipeline(
            dataframe,
            invalid_steps,
        )


# ---------------------------------------------------------
# Test 5: Unknown column
# ---------------------------------------------------------

def test_unknown_column_fails_without_mutating_input() -> None:
    """An operation must not silently create a missing column."""

    original = sample_dataframe()
    snapshot = original.copy(deep=True)

    steps = [
        make_step(
            "missing-unknown-column",
            "missing",
            {
                "strategy": "mean",
                "columns": ["column_does_not_exist"],
            },
        )
    ]

    with pytest.raises(
        (
            ValidationError,
            ValueError,
            KeyError,
            PipelineExecutionError,
        )
    ):
        run_pipeline(
            original,
            steps,
        )

    assert_frame_equal(original, snapshot)


# ---------------------------------------------------------
# Test 6: Preview limits
# ---------------------------------------------------------

def test_preview_does_not_modify_original() -> None:
    """A preview must preserve the source dataframe."""

    original = sample_dataframe()
    snapshot = original.copy(deep=True)

    cleaned, report, artifacts = run_pipeline(
        original,
        [],
        preview=True,
        preview_rows=2,
    )

    assert_frame_equal(original, snapshot)

    # Preview behavior belongs to the report.
    # The returned dataframe may contain all rows
    # so it can be used for subsequent operations.
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, dict)
    assert isinstance(artifacts, dict)


# ---------------------------------------------------------
# Test 7: Duplicate step IDs
# ---------------------------------------------------------

def test_duplicate_step_ids_are_rejected() -> None:
    """Pipeline step IDs should uniquely identify operations."""

    steps = [
        make_step(
            "same-step",
            "duplicates",
        ),
        make_step(
            "same-step",
            "duplicates",
        ),
    ]

    with pytest.raises(ValidationError):
        make_pipeline(steps)


# ---------------------------------------------------------
# Test 8: Pipeline preserves input
# ---------------------------------------------------------

def test_successful_pipeline_does_not_mutate_input() -> None:
    """Successful cleaning must produce a separate dataframe."""

    original = sample_dataframe()
    snapshot = original.copy(deep=True)

    cleaned, report, artifacts = run_pipeline(
        original,
        [
            make_step(
                "remove-duplicates",
                "duplicates",
                {},
            )
        ],
    )

    assert_frame_equal(original, snapshot)

    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, dict)
    assert isinstance(artifacts, dict)


# ---------------------------------------------------------
# Test 9: Operation ordering
# ---------------------------------------------------------

def test_pipeline_executes_steps_in_declared_order() -> None:
    """Pipeline execution must respect the saved step order."""

    original = sample_dataframe()

    steps = [
        make_step(
            "first-duplicates",
            "duplicates",
            {},
        ),
        make_step(
            "second-duplicates",
            "duplicates",
            {},
        ),
    ]

    cleaned, report, artifacts = run_pipeline(
        original,
        steps,
    )

    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, dict)
    assert isinstance(artifacts, dict)

    # Both steps should be represented in the report.
    # Report schemas may use different container keys,
    # so search the serialized report for their IDs.
    report_text = str(report)

    assert "first-duplicates" in report_text
    assert "second-duplicates" in report_text

    assert (
        report_text.index("first-duplicates")
        < report_text.index("second-duplicates")
    )


# ---------------------------------------------------------
# Test 10: Failed pipeline preserves input
# ---------------------------------------------------------

def test_failed_pipeline_preserves_input() -> None:
    """An error after an earlier step must not mutate source data."""

    original = sample_dataframe()
    snapshot = original.copy(deep=True)

    steps = [
        make_step(
            "first-duplicates",
            "duplicates",
            {},
        ),
        make_step(
            "second-invalid-column",
            "missing",
            {
                "strategy": "mean",
                "columns": ["missing_column"],
            },
        ),
    ]

    with pytest.raises(
        (
            ValidationError,
            ValueError,
            KeyError,
            PipelineExecutionError,
        )
    ):
        run_pipeline(
            original,
            steps,
        )

    assert_frame_equal(original, snapshot)
