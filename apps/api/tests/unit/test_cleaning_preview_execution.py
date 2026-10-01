
"""NUMPA v2 — Cleaning pipeline preview and execution consistency.

Verify that preview and execution produce consistent cleaned data.

The preview_rows parameter controls how many rows appear in the
preview report; it must not change the cleaning result itself.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


def sample_dataframe() -> pd.DataFrame:
    """Create data with missing values and duplicate records."""

    return pd.DataFrame(
        {
            "customer_id": [101, 101, 102, 103, 103, 104],
            "age": [20.0, None, 30.0, 40.0, 40.0, 50.0],
            "city": [
                "Surat",
                "Surat",
                "Rajkot",
                "Ahmedabad",
                "Ahmedabad",
                "Vadodara",
            ],
        }
    )


def missing_step(
    step_id: str = "fill-age",
    *,
    enabled: bool = True,
) -> dict:
    """Create an approved constant-imputation step."""

    return {
        "step_id": step_id,
        "operation": "missing_values",
        "parameters": {
            "strategy": "constant",
            "columns": ["age"],
            "fill_value": 20,
        },
        "enabled": enabled,
    }


def duplicate_step(
    step_id: str = "remove-duplicates",
    *,
    enabled: bool = True,
) -> dict:
    """Create a duplicate-removal step."""

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": {
            "strategy": "keep_first",
        },
        "enabled": enabled,
    }


def make_pipeline(
    steps: list[dict],
) -> CleaningPipelineConfig:
    """Build a validated pipeline."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Preview Consistency Pipeline",
            "steps": steps,
            "stop_on_error": True,
        }
    )


def compare_preview_and_execution(
    dataframe: pd.DataFrame,
    pipeline: CleaningPipelineConfig,
    *,
    preview_rows: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run preview and execution independently on identical inputs."""

    preview_data, preview_report, preview_artifacts = execute_pipeline(
        dataframe,
        pipeline,
        preview=True,
        preview_rows=preview_rows,
    )

    executed_data, execution_report, execution_artifacts = execute_pipeline(
        dataframe,
        pipeline,
        preview=False,
    )

    # The pipeline must not use different cleaning rules for preview.
    assert_frame_equal(
        preview_data,
        executed_data,
    )

    # Both actions must produce a report.
    assert preview_report is not None
    assert execution_report is not None

    return preview_data, executed_data


def test_duplicate_preview_matches_execution() -> None:
    """Duplicate removal must produce the same rows in both modes."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            duplicate_step(),
        ]
    )

    preview, executed = compare_preview_and_execution(
        dataframe,
        pipeline,
    )

    # The two exact duplicate records for customer 103 collapse to one.
    assert len(executed) == 5


def test_missing_value_preview_matches_execution() -> None:
    """Constant imputation must be identical in both modes."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(),
        ]
    )

    preview, executed = compare_preview_and_execution(
        dataframe,
        pipeline,
    )

    assert executed["age"].isna().sum() == 0
    assert executed.loc[1, "age"] == 20.0


def test_combined_pipeline_preview_matches_execution() -> None:
    """Preview must match execution across multiple cleaning steps."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(),
            duplicate_step(),
        ]
    )

    preview, executed = compare_preview_and_execution(
        dataframe,
        pipeline,
    )

    # Imputation creates an additional exact duplicate.
    assert executed["customer_id"].tolist() == [
        101,
        102,
        103,
        104,
    ]


def test_reversed_pipeline_preview_matches_execution() -> None:
    """Preview must respect the user's chosen step order."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            duplicate_step(),
            missing_step(),
        ]
    )

    preview, executed = compare_preview_and_execution(
        dataframe,
        pipeline,
    )

    # The first two records become identical only after deduplication.
    assert len(executed) == 5


@pytest.mark.parametrize(
    "preview_rows",
    [1, 2, 3, 10],
)
def test_preview_row_limit_does_not_change_cleaning(
    preview_rows: int,
) -> None:
    """Preview display limits must not alter the cleaned dataset."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(),
            duplicate_step(),
        ]
    )

    compare_preview_and_execution(
        dataframe,
        pipeline,
        preview_rows=preview_rows,
    )


def test_disabled_step_preview_matches_execution() -> None:
    """Disabled steps must be skipped consistently in both modes."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(enabled=False),
            duplicate_step(),
        ]
    )

    preview, executed = compare_preview_and_execution(
        dataframe,
        pipeline,
    )

    assert pd.isna(executed.loc[1, "age"])


def test_empty_pipeline_preview_matches_execution() -> None:
    """An empty pipeline must return the unchanged dataset."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline([])

    preview, executed = compare_preview_and_execution(
        dataframe,
        pipeline,
    )

    assert_frame_equal(preview, dataframe)
    assert_frame_equal(executed, dataframe)


def test_preview_does_not_modify_uploaded_dataset() -> None:
    """Preview must never modify the original uploaded data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = make_pipeline(
        [
            missing_step(),
            duplicate_step(),
        ]
    )

    execute_pipeline(
        dataframe,
        pipeline,
        preview=True,
        preview_rows=3,
    )

    assert_frame_equal(dataframe, original)


def test_execution_does_not_modify_uploaded_dataset() -> None:
    """Full execution must never modify the original uploaded data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = make_pipeline(
        [
            missing_step(),
            duplicate_step(),
        ]
    )

    execute_pipeline(
        dataframe,
        pipeline,
        preview=False,
    )

    assert_frame_equal(dataframe, original)


def test_preview_is_deterministic() -> None:
    """Two previews of the same pipeline must produce identical data."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(),
            duplicate_step(),
        ]
    )

    first, first_report, first_artifacts = execute_pipeline(
        dataframe,
        pipeline,
        preview=True,
        preview_rows=3,
    )

    second, second_report, second_artifacts = execute_pipeline(
        dataframe,
        pipeline,
        preview=True,
        preview_rows=3,
    )

    assert_frame_equal(first, second)


def test_non_default_indices_match_between_modes() -> None:
    """Preview and execution must retain the same surviving row indices."""

    dataframe = sample_dataframe()

    dataframe.index = [10, 20, 30, 40, 50, 60]

    pipeline = make_pipeline(
        [
            missing_step(),
            duplicate_step(),
        ]
    )

    preview, executed = compare_preview_and_execution(
        dataframe,
        pipeline,
    )

    assert preview.index.tolist() == executed.index.tolist()
    assert executed.index.tolist() == [10, 30, 40, 60]
