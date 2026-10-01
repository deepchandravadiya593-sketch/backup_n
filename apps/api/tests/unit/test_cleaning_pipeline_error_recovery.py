
"""NUMPA v2 - Cleaning pipeline error recovery tests.

Verify that:
1. Unknown operations are rejected.
2. Invalid duplicate configurations cannot execute successfully.
3. Invalid pipelines preserve the source dataset.
4. Runtime operation failures are surfaced.
5. Runtime failures preserve the source dataset.
6. Valid pipelines can run after a failed execution.
7. Repeated failures do not corrupt later executions.
8. Successful results survive subsequent failures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pydantic import ValidationError

import app.services.cleaning.pipeline as pipeline_module

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


# ---------------------------------------------------------
# Shared test data
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Create a dataset containing duplicate records."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 103, 104],
            "age": [24, 31, 38, 38, 45],
            "city": [
                "Surat",
                "Rajkot",
                "Ahmedabad",
                "Ahmedabad",
                "Vadodara",
            ],
            "spend": [
                1200.0,
                1800.0,
                2400.0,
                2400.0,
                3100.0,
            ],
        }
    )


def valid_pipeline() -> CleaningPipelineConfig:
    """Create a valid duplicate-removal pipeline."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Recovery Test Pipeline",
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


def execute_valid(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Execute the valid pipeline and return cleaned data."""

    cleaned, _report, _artifacts = execute_pipeline(
        dataframe,
        valid_pipeline(),
    )

    return cleaned


def assert_expected_cleaning(
    cleaned: pd.DataFrame,
) -> None:
    """Check that duplicate removal retains first occurrences."""

    expected = sample_dataframe().drop_duplicates(
        keep="first",
    )

    assert_frame_equal(
        cleaned.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def failing_operation(*args, **kwargs):
    """Simulate an operation failing during execution."""

    raise RuntimeError(
        "Simulated cleaning operation failure"
    )


def execute_with_simulated_failure(
    dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temporarily replace the internal operation executor."""

    # The context restores the original executor immediately
    # after this simulated failure.
    with monkeypatch.context() as patch:
        patch.setattr(
            pipeline_module,
            "_execute_step",
            failing_operation,
        )

        execute_pipeline(
            dataframe,
            valid_pipeline(),
        )


# ---------------------------------------------------------
# Test 1: Unknown operations are rejected
# ---------------------------------------------------------

def test_unknown_operation_is_rejected() -> None:
    """An unsupported operation must fail validation."""

    with pytest.raises(ValidationError):
        CleaningPipelineConfig.model_validate(
            {
                "name": "Invalid Operation",
                "steps": [
                    {
                        "step_id": "invalid-step",
                        "operation": "unknown_operation",
                        "parameters": {},
                    }
                ],
            }
        )


# ---------------------------------------------------------
# Test 2: Invalid duplicate configuration is rejected
# ---------------------------------------------------------

def test_invalid_duplicate_configuration_is_rejected() -> None:
    """An unsupported duplicate strategy must be rejected."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    invalid_definition = {
        "name": "Invalid Duplicate Configuration",
        "steps": [
            {
                "step_id": "invalid-duplicates",
                "operation": "duplicates",
                "parameters": {
                    "strategy": "unsupported_keep_mode",
                },
            }
        ],
    }

    try:
        # Some pipelines validate operation settings at creation.
        invalid_pipeline = CleaningPipelineConfig.model_validate(
            invalid_definition
        )

    except ValidationError:
        # Schema-level rejection is acceptable.
        assert_frame_equal(dataframe, original)
        return

    # Otherwise, the operation must reject the invalid strategy.
    with pytest.raises(
        (ValidationError, ValueError, TypeError, RuntimeError)
    ):
        execute_pipeline(
            dataframe,
            invalid_pipeline,
        )

    assert_frame_equal(dataframe, original)


# ---------------------------------------------------------
# Test 3: Validation errors preserve source data
# ---------------------------------------------------------

def test_invalid_pipeline_preserves_source() -> None:
    """Failed pipeline validation must not change input data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    with pytest.raises(ValidationError):
        CleaningPipelineConfig.model_validate(
            {
                "name": "Invalid Pipeline",
                "steps": [
                    {
                        "step_id": "bad-step",
                        "operation": "not_a_real_operation",
                        "parameters": {},
                    }
                ],
            }
        )

    assert_frame_equal(
        dataframe,
        original,
    )


# ---------------------------------------------------------
# Test 4: Runtime failures are surfaced
# ---------------------------------------------------------

def test_runtime_operation_failure_is_surfaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline must not silently report a failed step as successful."""

    dataframe = sample_dataframe()

    with pytest.raises(Exception) as error:
        execute_with_simulated_failure(
            dataframe,
            monkeypatch,
        )

    # Either the original error or a pipeline-specific
    # error wrapper is acceptable.
    assert error.value is not None


# ---------------------------------------------------------
# Test 5: Runtime failures preserve source data
# ---------------------------------------------------------

def test_runtime_failure_does_not_mutate_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed operation must preserve the uploaded dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    with pytest.raises(Exception):
        execute_with_simulated_failure(
            dataframe,
            monkeypatch,
        )

    assert_frame_equal(
        dataframe,
        original,
    )


# ---------------------------------------------------------
# Test 6: Recovery after failure
# ---------------------------------------------------------

def test_valid_pipeline_succeeds_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed execution must not prevent a later valid run."""

    dataframe = sample_dataframe()

    with pytest.raises(Exception):
        execute_with_simulated_failure(
            dataframe,
            monkeypatch,
        )

    # The temporary patch has now been removed.
    cleaned = execute_valid(
        dataframe,
    )

    assert_expected_cleaning(cleaned)


# ---------------------------------------------------------
# Test 7: Recovery after repeated failures
# ---------------------------------------------------------

def test_repeated_failures_do_not_corrupt_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated failures must not contaminate later executions."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    for _ in range(3):
        with pytest.raises(Exception):
            execute_with_simulated_failure(
                dataframe,
                monkeypatch,
            )

    assert_frame_equal(
        dataframe,
        original,
    )

    cleaned = execute_valid(
        dataframe,
    )

    assert_expected_cleaning(cleaned)


# ---------------------------------------------------------
# Test 8: Earlier success survives a later failure
# ---------------------------------------------------------

def test_successful_result_survives_later_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later failure must not change an earlier cleaned result."""

    dataframe = sample_dataframe()

    cleaned = execute_valid(
        dataframe,
    )

    successful_snapshot = cleaned.copy(deep=True)

    with pytest.raises(Exception):
        execute_with_simulated_failure(
            dataframe,
            monkeypatch,
        )

    assert_frame_equal(
        cleaned,
        successful_snapshot,
    )

    assert_expected_cleaning(cleaned)

