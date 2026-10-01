"""NUMPA v2 — End-to-end cleaning engine regression tests.

Verify the complete in-memory cleaning workflow using the
existing pipeline engine. HTTP upload and database persistence
are outside the scope of this file.
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


@pytest.fixture
def uploaded_dataset() -> pd.DataFrame:
    """Simulate a dataset loaded from an uploaded CSV or Excel file."""

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


def make_pipeline(
    *,
    stop_on_error: bool = True,
) -> CleaningPipelineConfig:
    """Configure missing-value replacement followed by deduplication."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Customer Data Cleaning",
            "stop_on_error": stop_on_error,
            "steps": [
                {
                    "step_id": "fill-age",
                    "operation": "missing_values",
                    "parameters": {
                        "strategy": "constant",
                        "columns": ["age"],
                        "fill_value": 20,
                    },
                    "enabled": True,
                },
                {
                    "step_id": "remove-duplicates",
                    "operation": "duplicates",
                    "parameters": {
                        "strategy": "keep_first",
                    },
                    "enabled": True,
                },
            ],
        }
    )


def test_complete_cleaning_workflow(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """The full pipeline must produce the expected cleaned records."""

    cleaned, report, artifacts = execute_pipeline(
        uploaded_dataset,
        make_pipeline(),
    )

    assert cleaned["customer_id"].tolist() == [101, 102, 103]
    assert cleaned["age"].tolist() == [20.0, 30.0, 40.0]
    assert cleaned["age"].isna().sum() == 0
    assert len(cleaned) == 3


def test_original_upload_is_preserved(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """Cleaning must not modify the uploaded dataset."""

    original = uploaded_dataset.copy(deep=True)

    execute_pipeline(uploaded_dataset, make_pipeline())

    assert_frame_equal(uploaded_dataset, original)


def test_pipeline_produces_audit_report(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """The workflow must return an audit report."""

    cleaned, report, artifacts = execute_pipeline(
        uploaded_dataset,
        make_pipeline(),
    )

    assert isinstance(report, dict)
    assert report


def test_both_operations_are_recorded(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """The audit report must identify both completed steps."""

    cleaned, report, artifacts = execute_pipeline(
        uploaded_dataset,
        make_pipeline(),
    )

    def find_completed_steps(value: object) -> set[str]:
        """Recursively locate completed pipeline-step records."""

        found: set[str] = set()

        if isinstance(value, dict):
            if (
                value.get("status") == "completed"
                and isinstance(value.get("step_id"), str)
            ):
                found.add(value["step_id"])

            for item in value.values():
                found.update(find_completed_steps(item))

        elif isinstance(value, list):
            for item in value:
                found.update(find_completed_steps(item))

        return found

    assert {
        "fill-age",
        "remove-duplicates",
    }.issubset(find_completed_steps(report))


def test_preview_does_not_change_cleaned_result(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """Preview mode must not change the final cleaned DataFrame."""

    pipeline = make_pipeline()

    normal, normal_report, normal_artifacts = execute_pipeline(
        uploaded_dataset,
        pipeline,
        preview=False,
    )

    preview, preview_report, preview_artifacts = execute_pipeline(
        uploaded_dataset,
        pipeline,
        preview=True,
        preview_rows=2,
    )

    assert_frame_equal(normal, preview)


def test_saved_pipeline_executes_identically(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """JSON serialization must preserve execution behavior."""

    original = make_pipeline()

    restored = CleaningPipelineConfig.model_validate_json(
        original.model_dump_json()
    )

    first, first_report, first_artifacts = execute_pipeline(
        uploaded_dataset,
        original,
    )

    second, second_report, second_artifacts = execute_pipeline(
        uploaded_dataset,
        restored,
    )

    assert_frame_equal(first, second)


def test_repeated_execution_is_deterministic(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """Identical inputs and configurations must yield identical results."""

    pipeline = make_pipeline()

    first, first_report, first_artifacts = execute_pipeline(
        uploaded_dataset,
        pipeline,
    )

    second, second_report, second_artifacts = execute_pipeline(
        uploaded_dataset,
        pipeline,
    )

    assert_frame_equal(first, second)
    assert first_report == second_report
    assert first_artifacts == second_artifacts


def test_invalid_step_does_not_damage_upload(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """A failing operation must preserve the uploaded dataset."""

    original = uploaded_dataset.copy(deep=True)
    pipeline = make_pipeline()

    # Make the second step fail during execution.
    pipeline.steps[1].parameters["columns"] = [
        "nonexistent_column"
    ]

    with pytest.raises(
        PipelineExecutionError,
        match="Unknown columns",
    ):
        execute_pipeline(
            uploaded_dataset,
            pipeline,
        )

    assert_frame_equal(uploaded_dataset, original)


def test_continue_on_error_keeps_successful_cleaning(
    uploaded_dataset: pd.DataFrame,
) -> None:
    """A failed second step must not undo successful imputation."""

    pipeline = make_pipeline(stop_on_error=False)

    # The first step is valid; the second step will fail.
    pipeline.steps[1].parameters["columns"] = [
        "nonexistent_column"
    ]

    cleaned, report, artifacts = execute_pipeline(
        uploaded_dataset,
        pipeline,
    )

    assert len(cleaned) == 5
    assert cleaned["age"].isna().sum() == 0
    assert cleaned.loc[1, "age"] == 20


def test_cleaned_dataset_can_be_exported(
    uploaded_dataset: pd.DataFrame,
    tmp_path,
) -> None:
    """The cleaned DataFrame must survive a CSV export and reload."""

    cleaned, report, artifacts = execute_pipeline(
        uploaded_dataset,
        make_pipeline(),
    )

    export_path = tmp_path / "cleaned_customers.csv"

    # Use the same basic CSV format offered by the product.
    cleaned.to_csv(export_path, index=False)

    reloaded = pd.read_csv(export_path)

    assert len(reloaded) == 3
    assert reloaded["customer_id"].tolist() == [101, 102, 103]
    assert reloaded["age"].isna().sum() == 0