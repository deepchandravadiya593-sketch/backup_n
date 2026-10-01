"""NUMPA v2 — Cleaning preview boundaries and artifact safety.

Test preview limits, source-data protection, and isolation
of previously fitted pipeline artifacts.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


def sample_dataframe() -> pd.DataFrame:
    """Create a small dataset containing a duplicate record."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103],
            "age": [20, 30, 30, 40],
            "city": ["Surat", "Rajkot", "Rajkot", "Ahmedabad"],
        }
    )


def empty_pipeline() -> CleaningPipelineConfig:
    """Create a valid pipeline without cleaning operations."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Preview Boundary Test",
            "steps": [],
            "stop_on_error": True,
        }
    )


def duplicate_pipeline() -> CleaningPipelineConfig:
    """Create a pipeline that removes exact duplicate records."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Duplicate Preview Test",
            "steps": [
                {
                    "step_id": "remove-duplicates",
                    "operation": "duplicates",
                    "parameters": {
                        "strategy": "keep_first",
                    },
                    "enabled": True,
                }
            ],
            "stop_on_error": True,
        }
    )


@pytest.mark.parametrize(
    "preview_rows",
    [-100, -1, 101, 1000],
)
def test_invalid_preview_limits_are_rejected(
    preview_rows: int,
) -> None:
    """Preview limits outside the supported range must fail."""

    dataframe = sample_dataframe()

    with pytest.raises(
        ValueError,
        match="preview_rows",
    ):
        execute_pipeline(
            dataframe,
            empty_pipeline(),
            preview=True,
            preview_rows=preview_rows,
        )


@pytest.mark.parametrize(
    "preview_rows",
    [0, 1, 10, 100],
)
def test_valid_preview_limits_are_accepted(
    preview_rows: int,
) -> None:
    """All supported preview limits must execute successfully."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        empty_pipeline(),
        preview=True,
        preview_rows=preview_rows,
    )

    # Preview size must not truncate the returned cleaned dataset.
    assert_frame_equal(cleaned, dataframe)
    assert isinstance(report, dict)
    assert isinstance(artifacts, dict)


def test_preview_does_not_mutate_uploaded_dataset() -> None:
    """Generating a preview must preserve the original dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    execute_pipeline(
        dataframe,
        duplicate_pipeline(),
        preview=True,
        preview_rows=2,
    )

    assert_frame_equal(dataframe, original)


def test_preview_and_execution_produce_same_cleaned_data() -> None:
    """Preview mode must not change the actual cleaning result."""

    dataframe = sample_dataframe()
    pipeline = duplicate_pipeline()

    preview_result, _, _ = execute_pipeline(
        dataframe,
        pipeline,
        preview=True,
        preview_rows=1,
    )

    execution_result, _, _ = execute_pipeline(
        dataframe,
        pipeline,
        preview=False,
    )

    assert_frame_equal(
        preview_result,
        execution_result,
    )

    assert len(execution_result) == 3


def test_incoming_artifacts_are_not_mutated() -> None:
    """Pipeline execution must not modify caller-owned artifacts."""

    dataframe = sample_dataframe()

    incoming_artifacts = {
        "existing-step": {
            "columns": ["age"],
            "statistics": {
                "mean": 30,
                "standard_deviation": 10,
            },
        }
    }

    original_artifacts = deepcopy(incoming_artifacts)

    execute_pipeline(
        dataframe,
        empty_pipeline(),
        fitted_artifacts=incoming_artifacts,
    )

    assert incoming_artifacts == original_artifacts


def test_returned_artifacts_are_independent() -> None:
    """Modifying returned artifacts must not affect the caller."""

    dataframe = sample_dataframe()

    incoming_artifacts = {
        "existing-step": {
            "columns": ["age"],
        }
    }

    cleaned, report, returned_artifacts = execute_pipeline(
        dataframe,
        empty_pipeline(),
        fitted_artifacts=incoming_artifacts,
    )

    returned_artifacts["existing-step"]["columns"].append(
        "customer_id"
    )

    assert incoming_artifacts["existing-step"]["columns"] == [
        "age"
    ]


def test_separate_executions_have_independent_artifacts() -> None:
    """Artifacts returned by one run must not leak into another."""

    dataframe = sample_dataframe()

    incoming_artifacts = {
        "existing-step": {
            "columns": ["age"],
        }
    }

    _, _, first_artifacts = execute_pipeline(
        dataframe,
        empty_pipeline(),
        fitted_artifacts=incoming_artifacts,
    )

    _, _, second_artifacts = execute_pipeline(
        dataframe,
        empty_pipeline(),
        fitted_artifacts=incoming_artifacts,
    )

    first_artifacts["existing-step"]["columns"].append(
        "city"
    )

    assert second_artifacts["existing-step"]["columns"] == [
        "age"
    ]


def test_invalid_preview_preserves_input_and_artifacts() -> None:
    """Validation failures must not modify either caller-owned input."""

    dataframe = sample_dataframe()
    original_dataframe = dataframe.copy(deep=True)

    incoming_artifacts = {
        "existing-step": {
            "columns": ["age"],
        }
    }

    original_artifacts = deepcopy(incoming_artifacts)

    with pytest.raises(ValueError):
        execute_pipeline(
            dataframe,
            duplicate_pipeline(),
            fitted_artifacts=incoming_artifacts,
            preview=True,
            preview_rows=101,
        )

    assert_frame_equal(
        dataframe,
        original_dataframe,
    )

    assert incoming_artifacts == original_artifacts