"""NUMPA v2 — Continue-on-error pipeline tests.

Test recovery from invalid cleaning steps while protecting
the uploaded dataset and preserving successful operations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    PipelineExecutionError,
    execute_pipeline,
)


def sample_dataframe() -> pd.DataFrame:
    """Create data with missing values and duplicate records."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103],
            "age": [20.0, None, None, 40.0],
            "city": [
                "Surat",
                "Rajkot",
                "Rajkot",
                "Ahmedabad",
            ],
        }
    )


def fill_step(step_id: str = "fill-age") -> dict:
    """Create a valid constant-imputation step."""

    return {
        "step_id": step_id,
        "operation": "missing_values",
        "parameters": {
            "strategy": "constant",
            "columns": ["age"],
            "fill_value": 25,
        },
        "enabled": True,
    }


def duplicate_step(step_id: str = "remove-duplicates") -> dict:
    """Create a valid duplicate-removal step."""

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": {
            "strategy": "keep_first",
        },
        "enabled": True,
    }


def failing_step(step_id: str = "invalid-duplicates") -> dict:
    """Create a step referencing a nonexistent dataset column."""

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": {
            "strategy": "keep_first",
            "columns": ["nonexistent_column"],
        },
        "enabled": True,
    }


def run_pipeline(
    dataframe: pd.DataFrame,
    steps: list[dict],
    *,
    stop_on_error: bool = False,
):
    """Execute an ordered pipeline with explicit error handling."""

    config = CleaningPipelineConfig.model_validate(
        {
            "name": "Continue on Error Test",
            "steps": steps,
            "stop_on_error": stop_on_error,
        }
    )

    return execute_pipeline(dataframe, config)


def test_failed_step_does_not_stop_pipeline() -> None:
    """Continue-on-error mode must return a result."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_pipeline(
        dataframe,
        [failing_step()],
    )

    assert_frame_equal(cleaned, dataframe)


def test_successful_step_runs_after_failure() -> None:
    """A valid step following a failed step must still execute."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_pipeline(
        dataframe,
        [
            failing_step(),
            fill_step(),
        ],
    )

    assert cleaned["age"].isna().sum() == 0
    assert cleaned.loc[1, "age"] == 25
    assert len(cleaned) == 4


def test_successful_step_survives_later_failure() -> None:
    """A later failure must not undo an earlier successful step."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_pipeline(
        dataframe,
        [
            fill_step(),
            failing_step(),
        ],
    )

    assert cleaned["age"].isna().sum() == 0
    assert cleaned.loc[1, "age"] == 25


def test_valid_steps_run_on_both_sides_of_failure() -> None:
    """Successful steps before and after a failure must execute."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_pipeline(
        dataframe,
        [
            fill_step(),
            failing_step(),
            duplicate_step(),
        ],
    )

    assert cleaned["customer_id"].tolist() == [
        101,
        102,
        103,
    ]

    assert cleaned["age"].tolist() == [
        20.0,
        25.0,
        40.0,
    ]


def test_multiple_failed_steps_do_not_stop_execution() -> None:
    """Multiple invalid operations must not prevent later work."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_pipeline(
        dataframe,
        [
            failing_step("failure-one"),
            failing_step("failure-two"),
            fill_step(),
        ],
    )

    assert len(cleaned) == 4
    assert cleaned["age"].isna().sum() == 0


def test_failed_step_preserves_uploaded_dataset() -> None:
    """A failed operation must never modify the source DataFrame."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_pipeline(
        dataframe,
        [
            fill_step(),
            failing_step(),
            duplicate_step(),
        ],
    )

    assert_frame_equal(dataframe, original)


def test_stop_on_error_raises_pipeline_exception() -> None:
    """Strict execution must stop at the first failed step."""

    dataframe = sample_dataframe()

    with pytest.raises(
        PipelineExecutionError,
        match="Unknown columns",
    ):
        run_pipeline(
            dataframe,
            [
                failing_step(),
                fill_step(),
            ],
            stop_on_error=True,
        )


def test_disabled_invalid_step_is_ignored() -> None:
    """A disabled step must not execute, even if its columns are invalid."""

    dataframe = sample_dataframe()

    invalid = failing_step()
    invalid["enabled"] = False

    cleaned, report, artifacts = run_pipeline(
        dataframe,
        [
            invalid,
            fill_step(),
        ],
        stop_on_error=True,
    )

    assert cleaned["age"].isna().sum() == 0


def test_recovery_is_deterministic() -> None:
    """Repeating the same failing pipeline must produce the same data."""

    dataframe = sample_dataframe()

    steps = [
        fill_step(),
        failing_step(),
        duplicate_step(),
    ]

    first, first_report, first_artifacts = run_pipeline(
        dataframe,
        steps,
    )

    second, second_report, second_artifacts = run_pipeline(
        dataframe,
        steps,
    )

    assert_frame_equal(first, second)
    assert first_report == second_report
    assert first_artifacts == second_artifacts


def test_failure_report_contains_step_identifier() -> None:
    """The audit report must identify the failed operation."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_pipeline(
        dataframe,
        [
            failing_step("broken-column-step"),
        ],
    )

    # Inspect the report as structured data without depending on
    # the exact name of its step-report collection.
    def contains_failure(value: object) -> bool:
        """Recursively locate the expected failed-step record."""

        if isinstance(value, dict):
            if (
                value.get("step_id") == "broken-column-step"
                and value.get("status") == "failed"
            ):
                return True

            return any(
                contains_failure(item)
                for item in value.values()
            )

        if isinstance(value, list):
            return any(
                contains_failure(item)
                for item in value
            )

        return False

    assert contains_failure(report)