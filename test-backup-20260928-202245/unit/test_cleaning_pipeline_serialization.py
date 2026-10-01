
"""NUMPA v2 — Cleaning pipeline serialization tests.

Verify that cleaning pipelines can be saved, restored,
modified independently, and executed consistently.
"""

from __future__ import annotations

import json

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
    """Create a dataset suitable for testing saved pipelines."""

    return pd.DataFrame(
        {
            "customer_id": [101, 101, 102, 103, 103],
            "age": [20.0, None, 30.0, 40.0, 40.0],
            "city": [
                "Surat",
                "Surat",
                "Rajkot",
                "Ahmedabad",
                "Ahmedabad",
            ],
        }
    )


def sample_pipeline() -> CleaningPipelineConfig:
    """Create a valid, ordered cleaning pipeline."""

    return CleaningPipelineConfig(
        name="Customer Data Cleaning",
        steps=[
            PipelineStep(
                step_id="fill-age",
                operation="missing_values",
                parameters={
                    "strategy": "constant",
                    "columns": ["age"],
                    "fill_value": 20,
                },
                enabled=True,
            ),
            PipelineStep(
                step_id="remove-duplicates",
                operation="duplicates",
                parameters={
                    "strategy": "keep_first",
                    "columns": [
                        "customer_id",
                        "age",
                        "city",
                    ],
                },
                enabled=True,
            ),
        ],
        stop_on_error=True,
    )


def run_pipeline(
    dataframe: pd.DataFrame,
    pipeline: CleaningPipelineConfig,
) -> pd.DataFrame:
    """Execute a pipeline and return its cleaned dataset."""

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    return cleaned


def test_pipeline_serializes_to_dictionary() -> None:
    """A saved pipeline must contain its essential settings."""

    pipeline = sample_pipeline()

    saved = pipeline.model_dump()

    assert saved["name"] == "Customer Data Cleaning"
    assert saved["stop_on_error"] is True
    assert len(saved["steps"]) == 2


def test_pipeline_serializes_to_valid_json() -> None:
    """Pipeline JSON must be compatible with standard JSON tools."""

    pipeline = sample_pipeline()

    saved_json = pipeline.model_dump_json()
    parsed = json.loads(saved_json)

    assert parsed["name"] == pipeline.name
    assert len(parsed["steps"]) == 2


def test_pipeline_restores_from_dictionary() -> None:
    """A dictionary round-trip must preserve the entire configuration."""

    original = sample_pipeline()

    restored = CleaningPipelineConfig.model_validate(
        original.model_dump()
    )

    assert restored.model_dump() == original.model_dump()


def test_pipeline_restores_from_json() -> None:
    """A JSON round-trip must preserve the entire configuration."""

    original = sample_pipeline()

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    assert restored.model_dump() == original.model_dump()


def test_step_order_survives_serialization() -> None:
    """Saving and loading must not reorder cleaning operations."""

    original = sample_pipeline()

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    assert [
        step.step_id for step in restored.steps
    ] == [
        "fill-age",
        "remove-duplicates",
    ]


def test_step_parameters_survive_serialization() -> None:
    """All operation-specific parameters must be retained."""

    original = sample_pipeline()

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    assert restored.steps[0].parameters == {
        "strategy": "constant",
        "columns": ["age"],
        "fill_value": 20,
    }

    assert restored.steps[1].parameters == {
        "strategy": "keep_first",
        "columns": [
            "customer_id",
            "age",
            "city",
        ],
    }


def test_disabled_step_survives_serialization() -> None:
    """Disabled operations must remain disabled after restoration."""

    original = sample_pipeline()
    original.steps[1].enabled = False

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    assert restored.steps[0].enabled is True
    assert restored.steps[1].enabled is False


def test_stop_on_error_survives_serialization() -> None:
    """The pipeline's error-handling preference must be retained."""

    original = sample_pipeline()
    original.stop_on_error = False

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    assert restored.stop_on_error is False


def test_restored_pipeline_executes_identically() -> None:
    """Restoring a pipeline must not change its cleaning results."""

    dataframe = sample_dataframe()
    original = sample_pipeline()

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    first_result = run_pipeline(dataframe, original)
    second_result = run_pipeline(dataframe, restored)

    assert_frame_equal(first_result, second_result)

    # Imputation creates one duplicate; another already exists.
    assert first_result["customer_id"].tolist() == [
        101,
        102,
        103,
    ]


def test_disabled_step_behavior_survives_restoration() -> None:
    """A restored disabled step must remain inactive during execution."""

    dataframe = sample_dataframe()
    original = sample_pipeline()
    original.steps[1].enabled = False

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    cleaned = run_pipeline(dataframe, restored)

    # Imputation runs, but duplicate removal is disabled.
    assert len(cleaned) == 5
    assert cleaned["age"].isna().sum() == 0


def test_restored_pipeline_is_independent() -> None:
    """Changing a restored pipeline must not modify the original."""

    original = sample_pipeline()

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    restored.name = "Modified Pipeline"
    restored.steps[0].parameters["fill_value"] = 99
    restored.steps[1].enabled = False

    assert original.name == "Customer Data Cleaning"
    assert original.steps[0].parameters["fill_value"] == 20
    assert original.steps[1].enabled is True


def test_duplicate_step_ids_are_rejected_after_loading() -> None:
    """Saved pipelines must still satisfy step-ID validation."""

    saved = sample_pipeline().model_dump()

    saved["steps"][1]["step_id"] = "fill-age"

    with pytest.raises(ValidationError):
        CleaningPipelineConfig.model_validate(saved)


def test_invalid_operation_is_rejected_after_loading() -> None:
    """A saved file must not introduce unsupported operations."""

    saved = sample_pipeline().model_dump()

    saved["steps"][0]["operation"] = "unknown_operation"

    with pytest.raises(ValidationError):
        CleaningPipelineConfig.model_validate(saved)


def test_obsolete_step_fields_are_rejected() -> None:
    """Unexpected step-level fields must not enter saved pipelines."""

    saved = sample_pipeline().model_dump()

    saved["steps"][1]["keep"] = "last"

    with pytest.raises(ValidationError):
        CleaningPipelineConfig.model_validate(saved)


def test_original_dataset_survives_restored_execution() -> None:
    """Running a restored pipeline must preserve the uploaded data."""

    dataframe = sample_dataframe()
    original_dataframe = dataframe.copy(deep=True)

    restored = CleaningPipelineConfig.model_validate_json(
        sample_pipeline().model_dump_json()
    )

    run_pipeline(dataframe, restored)

    assert_frame_equal(dataframe, original_dataframe)
