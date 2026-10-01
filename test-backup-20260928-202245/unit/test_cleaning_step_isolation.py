
"""NUMPA v2 — Cleaning pipeline step isolation tests.

Verify that:
- Each enabled step receives the previous step's output.
- Disabled steps do not affect the result.
- Reordering steps can change the result.
- Different pipelines do not share execution state.
- Original uploaded data remains unchanged.
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
    """Create a dataset containing missing and duplicate values."""

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


def missing_step(
    step_id: str,
    *,
    fill_value: int = 20,
    enabled: bool = True,
) -> dict:
    """Build an explicitly configured missing-value operation."""

    return {
        "step_id": step_id,
        "operation": "missing_values",
        "parameters": {
            "strategy": "constant",
            "columns": ["age"],
            "fill_value": fill_value,
        },
        "enabled": enabled,
    }


def duplicate_step(
    step_id: str,
    *,
    columns: list[str] | None = None,
    enabled: bool = True,
) -> dict:
    """Build a duplicate-removal operation."""

    parameters = {
        "strategy": "keep_first",
    }

    if columns is not None:
        parameters["columns"] = columns

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": parameters,
        "enabled": enabled,
    }


def run_pipeline(
    dataframe: pd.DataFrame,
    steps: list[dict],
) -> pd.DataFrame:
    """Execute an independently validated cleaning pipeline."""

    config = CleaningPipelineConfig.model_validate(
        {
            "name": "Step Isolation Test",
            "steps": steps,
            "stop_on_error": True,
        }
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        config,
    )

    return cleaned


def test_second_step_receives_first_step_output() -> None:
    """Duplicate removal must detect duplicates created by imputation."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step("fill-age"),
            duplicate_step("remove-duplicates"),
        ],
    )

    # Filling the missing age creates a duplicate for customer 101.
    # Customer 103 already has two identical records.
    assert cleaned["customer_id"].tolist() == [101, 102, 103]
    assert cleaned["age"].tolist() == [20.0, 30.0, 40.0]


def test_reversing_steps_changes_result() -> None:
    """Removing duplicates before imputation can leave new duplicates."""

    dataframe = sample_dataframe()

    imputation_first = run_pipeline(
        dataframe,
        [
            missing_step("fill-age"),
            duplicate_step("remove-duplicates"),
        ],
    )

    duplicates_first = run_pipeline(
        dataframe,
        [
            duplicate_step("remove-duplicates"),
            missing_step("fill-age"),
        ],
    )

    assert len(imputation_first) == 3
    assert len(duplicates_first) == 4


def test_disabled_first_step_does_not_change_second_step() -> None:
    """A disabled imputation step must not create additional duplicates."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step(
                "disabled-fill",
                enabled=False,
            ),
            duplicate_step("remove-duplicates"),
        ],
    )

    assert len(cleaned) == 4
    assert pd.isna(cleaned.loc[1, "age"])


def test_disabled_second_step_preserves_duplicate_rows() -> None:
    """A disabled duplicate operation must not remove any records."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step("fill-age"),
            duplicate_step(
                "disabled-duplicates",
                enabled=False,
            ),
        ],
    )

    assert len(cleaned) == 5
    assert cleaned["age"].isna().sum() == 0


def test_two_disabled_steps_preserve_original_data() -> None:
    """A pipeline containing only disabled steps must do nothing."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step(
                "disabled-fill",
                enabled=False,
            ),
            duplicate_step(
                "disabled-duplicates",
                enabled=False,
            ),
        ],
    )

    assert_frame_equal(cleaned, dataframe)


def test_independent_pipelines_do_not_share_results() -> None:
    """Running one pipeline must not affect a separate pipeline."""

    dataframe = sample_dataframe()

    imputation_result = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-age",
                fill_value=25,
            ),
        ],
    )

    duplicate_result = run_pipeline(
        dataframe,
        [
            duplicate_step("remove-duplicates"),
        ],
    )

    # The first pipeline imputes age but does not remove rows.
    assert len(imputation_result) == 5
    assert imputation_result.loc[1, "age"] == 25

    # The second pipeline removes exact duplicates but does not impute.
    assert len(duplicate_result) == 4
    assert pd.isna(duplicate_result.loc[1, "age"])


def test_different_parameters_produce_independent_results() -> None:
    """Changing one pipeline's fill value must not affect another."""

    dataframe = sample_dataframe()

    first = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-with-20",
                fill_value=20,
            ),
        ],
    )

    second = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-with-25",
                fill_value=25,
            ),
        ],
    )

    assert first.loc[1, "age"] == 20
    assert second.loc[1, "age"] == 25


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [
            missing_step("fill-age"),
        ],
        [
            duplicate_step("remove-duplicates"),
        ],
        [
            missing_step("fill-age"),
            duplicate_step("remove-duplicates"),
        ],
    ],
    ids=[
        "empty-pipeline",
        "missing-values-only",
        "duplicates-only",
        "combined-pipeline",
    ],
)
def test_pipeline_never_mutates_uploaded_data(
    steps: list[dict],
) -> None:
    """All supported pipeline combinations must protect source data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_pipeline(
        dataframe,
        steps,
    )

    assert_frame_equal(dataframe, original)


def test_repeated_combined_execution_is_deterministic() -> None:
    """Identical pipeline executions must produce identical results."""

    dataframe = sample_dataframe()

    steps = [
        missing_step("fill-age"),
        duplicate_step("remove-duplicates"),
    ]

    first = run_pipeline(dataframe, steps)
    second = run_pipeline(dataframe, steps)

    assert_frame_equal(first, second)


def test_original_row_indices_are_preserved() -> None:
    """Surviving rows must retain their original source-row indices."""

    dataframe = sample_dataframe()
    dataframe.index = [10, 20, 30, 40, 50]

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step("fill-age"),
            duplicate_step("remove-duplicates"),
        ],
    )

    assert cleaned.index.tolist() == [10, 30, 40]
