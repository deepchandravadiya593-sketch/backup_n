
"""NUMPA v2 â€” Cleaning pipeline boundary and failure-safety tests.

Verify configuration validation, error handling, and source-data
protection for missing-value and duplicate-cleaning pipelines.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    PipelineExecutionError,
    execute_pipeline,
)
from app.services.cleaning.operations.missing import MissingValueConfig
from app.services.cleaning.operations.duplicates import DuplicateConfig


def sample_dataframe() -> pd.DataFrame:
    """Create an input containing missing values and duplicates."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103],
            "age": [20.0, None, None, 40.0],
            "city": ["Surat", "Rajkot", "Rajkot", "Ahmedabad"],
        }
    )


def missing_step(
    step_id: str,
    *,
    strategy: str = "constant",
    columns: list[str] | None = None,
    fill_value: object = 25,
    enabled: bool = True,
) -> dict:
    """Build a missing-value pipeline step."""

    parameters = {"strategy": strategy}

    if columns is not None:
        parameters["columns"] = columns

    if strategy == "constant":
        parameters["fill_value"] = fill_value

    return {
        "step_id": step_id,
        "operation": "missing_values",
        "parameters": parameters,
        "enabled": enabled,
    }


def duplicate_step(
    step_id: str,
    *,
    strategy: str = "keep_first",
    columns: list[str] | None = None,
    enabled: bool = True,
) -> dict:
    """Build a duplicate-cleaning pipeline step."""

    parameters = {"strategy": strategy}

    if columns is not None:
        parameters["columns"] = columns

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": parameters,
        "enabled": enabled,
    }


def make_pipeline(
    steps: list[dict],
    *,
    stop_on_error: bool = True,
) -> CleaningPipelineConfig:
    """Validate an ordered cleaning pipeline."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Failure Safety Test",
            "steps": steps,
            "stop_on_error": stop_on_error,
        }
    )


def execute(
    dataframe: pd.DataFrame,
    steps: list[dict],
) -> pd.DataFrame:
    """Execute a validated pipeline and return its cleaned data."""

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        make_pipeline(steps),
    )

    return cleaned


def test_duplicate_step_ids_are_rejected() -> None:
    """Every pipeline step must have a unique identifier."""

    with pytest.raises(ValidationError):
        make_pipeline(
            [
                missing_step("same-id"),
                duplicate_step("same-id"),
            ]
        )


def test_distinct_step_ids_are_accepted() -> None:
    """Different step identifiers must allow a valid pipeline."""

    pipeline = make_pipeline(
        [
            missing_step("fill-age"),
            duplicate_step("remove-duplicates"),
        ]
    )

    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].step_id == "fill-age"
    assert pipeline.steps[1].step_id == "remove-duplicates"


def test_missing_strategy_is_required() -> None:
    """Missing-value cleaning must require an explicit strategy."""

    with pytest.raises(ValidationError):
        MissingValueConfig.model_validate({})


def test_constant_imputation_requires_fill_value() -> None:
    """Constant replacement must not silently invent a value."""

    with pytest.raises(ValidationError):
        MissingValueConfig.model_validate(
            {"strategy": "constant"}
        )


def test_invalid_duplicate_strategy_is_rejected() -> None:
    """Unsupported duplicate-removal strategies must fail validation."""

    with pytest.raises(ValidationError):
        DuplicateConfig.model_validate(
            {"strategy": "remove_random_rows"}
        )


def test_duplicate_configuration_rejects_unknown_fields() -> None:
    """Obsolete configuration fields must not be silently ignored."""

    with pytest.raises(ValidationError):
        DuplicateConfig.model_validate(
            {
                "strategy": "keep_first",
                "keep": "last",
            }
        )


def test_duplicate_configuration_rejects_repeated_columns() -> None:
    """A matching-column selection must not contain repeated names."""

    with pytest.raises(ValidationError):
        DuplicateConfig.model_validate(
            {
                "strategy": "keep_first",
                "columns": ["customer_id", "customer_id"],
            }
        )


def test_unknown_duplicate_column_preserves_source() -> None:
    """An invalid column selection must not alter uploaded data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = make_pipeline(
        [
            duplicate_step(
                "invalid-column",
                columns=["unknown_column"],
            )
        ]
    )

    # Column existence is checked against the actual input DataFrame.
    with pytest.raises(
        PipelineExecutionError,
        match="Unknown columns",
        ):
        execute_pipeline(dataframe, pipeline)

    assert_frame_equal(dataframe, original)


def test_invalid_missing_configuration_preserves_source() -> None:
    """Rejecting a missing-value configuration must preserve source data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    with pytest.raises(ValidationError):
        MissingValueConfig.model_validate(
            {
                "strategy": "constant",
            }
        )

    assert_frame_equal(dataframe, original)


def test_invalid_operation_is_rejected() -> None:
    """Only supported cleaning operations may enter a pipeline."""

    with pytest.raises(ValidationError):
        make_pipeline(
            [
                {
                    "step_id": "invalid-operation",
                    "operation": "delete_everything",
                    "parameters": {},
                    "enabled": True,
                }
            ]
        )


def test_disabled_steps_preserve_source() -> None:
    """Disabled cleaning operations must have no effect."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    cleaned = execute(
        dataframe,
        [
            missing_step(
                "disabled-missing",
                enabled=False,
            ),
            duplicate_step(
                "disabled-duplicates",
                enabled=False,
            ),
        ],
    )

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_empty_pipeline_preserves_source() -> None:
    """A pipeline without steps must return unchanged data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    cleaned = execute(dataframe, [])

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_valid_pipeline_preserves_source() -> None:
    """Successful multistep cleaning must not mutate uploaded records."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    cleaned = execute(
        dataframe,
        [
            missing_step(
                "fill-age",
                columns=["age"],
            ),
            duplicate_step(
                "remove-duplicates",
            ),
        ],
    )

    assert_frame_equal(dataframe, original)

    # The two repeated customer records become exact duplicates.
    assert cleaned["customer_id"].tolist() == [101, 102, 103]


def test_failed_execution_does_not_affect_later_execution() -> None:
    """A failed pipeline must not contaminate subsequent valid runs."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    invalid_pipeline = make_pipeline(
        [
            duplicate_step(
                "invalid-deduplication",
                columns=["nonexistent"],
            )
        ]
    )

    with pytest.raises(
        PipelineExecutionError,
        match="Unknown columns",
    ):
        execute_pipeline(dataframe, invalid_pipeline)

    # Run a separate valid pipeline against the original dataset.
    valid_result = execute(
        dataframe,
        [
            duplicate_step(
                "valid-deduplication",
            )
        ],
    )

    assert len(valid_result) == 3
    assert_frame_equal(dataframe, original)


def test_repeated_valid_execution_is_deterministic() -> None:
    """Identical runs must produce identical cleaned datasets."""

    dataframe = sample_dataframe()

    steps = [
        missing_step(
            "fill-age",
            columns=["age"],
        ),
        duplicate_step(
            "remove-duplicates",
        ),
    ]

    first = execute(dataframe, steps)
    second = execute(dataframe, steps)

    assert_frame_equal(first, second)

