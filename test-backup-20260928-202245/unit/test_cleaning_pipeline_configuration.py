
"""NUMPA v2 — Cleaning pipeline configuration tests.

Verify that pipeline definitions are validated correctly and
that user-approved parameters reach the cleaning operations.

These tests protect the contract between the frontend's visual
pipeline builder and the Python cleaning engine.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    PipelineStep,
    execute_pipeline,
)


def sample_dataframe() -> pd.DataFrame:
    """Create a small dataset with two identical records."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103],
            "city": [
                "Surat",
                "Rajkot",
                "Rajkot",
                "Ahmedabad",
            ],
            "spend": [1200, 1800, 1800, 2400],
        }
    )


def make_pipeline(
    steps: list[dict],
) -> CleaningPipelineConfig:
    """Validate a pipeline definition using the public model."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Configuration Test Pipeline",
            "steps": steps,
        }
    )


def duplicate_step(
    *,
    step_id: str = "remove-duplicates",
    strategy: str = "keep_first",
    enabled: bool = True,
) -> dict:
    """Build a correctly structured duplicate-removal step."""

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": {
            "strategy": strategy,
        },
        "enabled": enabled,
    }


def test_empty_pipeline_is_valid() -> None:
    """An empty pipeline should be a valid no-op definition."""

    pipeline = make_pipeline([])

    assert pipeline.steps == []
    assert pipeline.name == "Configuration Test Pipeline"


def test_supported_operations_are_accepted() -> None:
    """Verify that all eight operation names are supported.

    This test checks the allowed operation names independently
    of each operation's required configuration parameters.
    """

    # Read the operation names directly from PipelineStep's schema.
    # This avoids supplying invalid empty configurations to
    # operations that require parameters such as "strategy".
    operation_schema = PipelineStep.model_fields["operation"]

    supported_operations = set(
        operation_schema.annotation.__args__
    )

    expected_operations = {
        "missing_values",
        "duplicates",
        "encoding",
        "scaling",
        "outliers",
        "datatypes",
        "features",
        "text",
    }

    # Confirm that the model exposes exactly the supported operations.
    assert supported_operations == expected_operations


def test_unknown_operation_is_rejected() -> None:
    """The frontend must not submit unsupported operation names."""

    with pytest.raises(ValidationError):
        make_pipeline(
            [
                {
                    "step_id": "invalid-operation",
                    "operation": "unknown_operation",
                    "parameters": {},
                }
            ]
        )


def test_parameters_are_preserved() -> None:
    """User-approved settings must reach the operation unchanged."""

    pipeline = make_pipeline(
        [
            duplicate_step(
                strategy="keep_last",
            )
        ]
    )

    step = pipeline.steps[0]

    assert step.operation == "duplicates"
    assert step.parameters == {
        "strategy": "keep_last",
    }


def test_obsolete_keep_field_is_rejected() -> None:
    """The obsolete step-level 'keep' field must not be accepted."""

    with pytest.raises(ValidationError):
        make_pipeline(
            [
                {
                    "step_id": "invalid-duplicates",
                    "operation": "duplicates",
                    "keep": "first",
                }
            ]
        )


def test_duplicate_step_ids_are_rejected() -> None:
    """Step identifiers must be unique within one pipeline."""

    with pytest.raises(
        ValidationError,
        match="Pipeline step IDs must be unique",
    ):
        make_pipeline(
            [
                duplicate_step(
                    step_id="same-id",
                ),
                duplicate_step(
                    step_id="same-id",
                ),
            ]
        )


def test_distinct_step_ids_are_accepted() -> None:
    """Different step identifiers must remain independent."""

    pipeline = make_pipeline(
        [
            duplicate_step(
                step_id="first-step",
            ),
            duplicate_step(
                step_id="second-step",
            ),
        ]
    )

    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].step_id == "first-step"
    assert pipeline.steps[1].step_id == "second-step"


def test_pipeline_preserves_step_order() -> None:
    """The backend must preserve the user's visual step order."""

    pipeline = make_pipeline(
        [
            duplicate_step(
                step_id="step-a",
            ),
            duplicate_step(
                step_id="step-b",
            ),
            duplicate_step(
                step_id="step-c",
            ),
        ]
    )

    assert [
        step.step_id
        for step in pipeline.steps
    ] == [
        "step-a",
        "step-b",
        "step-c",
    ]


def test_disabled_step_preserves_source_rows() -> None:
    """A disabled duplicate-removal step must not remove records."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = make_pipeline(
        [
            duplicate_step(
                enabled=False,
            )
        ]
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_enabled_duplicate_step_uses_parameters() -> None:
    """The selected strategy must control which row survives."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            duplicate_step(
                strategy="keep_last",
            )
        ]
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    # The first occurrence of customer 102 is removed.
    expected = dataframe.iloc[
        [0, 2, 3]
    ].copy()

    assert_frame_equal(cleaned, expected)


def test_invalid_duplicate_strategy_fails_execution() -> None:
    """Reject invalid strategies during configuration or execution."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    definition = [
        duplicate_step(
            strategy="unsupported_strategy",
        )
    ]

    try:
        # Some pipeline implementations validate parameters here.
        pipeline = make_pipeline(definition)

    except ValidationError:
        # Rejection during pipeline creation is valid.
        assert_frame_equal(dataframe, original)
        return

    # Other implementations validate operation parameters here.
    with pytest.raises(ValidationError):
        execute_pipeline(
            dataframe,
            pipeline,
        )

    # A failed operation must never modify the source dataset.
    assert_frame_equal(dataframe, original)


def test_pipeline_configuration_does_not_mutate_parameters() -> None:
    """Model validation must not rewrite the caller's configuration."""

    definition = duplicate_step(
        strategy="remove_all",
    )

    original_parameters = definition[
        "parameters"
    ].copy()

    pipeline = make_pipeline([definition])

    assert definition["parameters"] == original_parameters

    assert pipeline.steps[0].parameters == {
        "strategy": "remove_all",
    }
