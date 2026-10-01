
"""NUMPA v2 — Cleaning pipeline idempotency tests.

Verify that completed cleaning pipelines produce stable results
when executed again on their own output.

Coverage:
- Duplicate-removal idempotency.
- Constant-imputation idempotency.
- Mean-imputation idempotency.
- Combined cleaning idempotency.
- Row-removal idempotency.
- Independent repeated executions.
- Source-data preservation.
- Stable results with non-default row indices.
"""

from __future__ import annotations

import pandas as pd

from pandas.testing import assert_frame_equal

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


def sample_dataframe() -> pd.DataFrame:
    """Return data containing missing values and duplicate records."""

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
    strategy: str,
    columns: list[str],
    *,
    fill_value: object = None,
) -> dict:
    """Build a valid missing-value cleaning step."""

    parameters = {
        "strategy": strategy,
        "columns": columns,
    }

    # Constant imputation requires an explicit replacement.
    if strategy == "constant":
        parameters["fill_value"] = fill_value

    return {
        "step_id": step_id,
        "operation": "missing_values",
        "parameters": parameters,
        "enabled": True,
    }


def duplicate_step(
    step_id: str,
    *,
    strategy: str = "keep_first",
    columns: list[str] | None = None,
) -> dict:
    """Build a valid duplicate-removal step."""

    parameters = {
        "strategy": strategy,
    }

    if columns is not None:
        parameters["columns"] = columns

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": parameters,
        "enabled": True,
    }


def make_pipeline(
    steps: list[dict],
) -> CleaningPipelineConfig:
    """Validate the complete pipeline definition."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Idempotency Test Pipeline",
            "steps": steps,
            "stop_on_error": True,
        }
    )


def run_pipeline(
    dataframe: pd.DataFrame,
    pipeline: CleaningPipelineConfig,
) -> pd.DataFrame:
    """Execute a pipeline and return the cleaned DataFrame."""

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    return cleaned


def assert_idempotent(
    dataframe: pd.DataFrame,
    pipeline: CleaningPipelineConfig,
) -> None:
    """Check that a second execution leaves the result unchanged."""

    # First execution produces the cleaned dataset.
    first = run_pipeline(
        dataframe,
        pipeline,
    )

    # Execute the same pipeline on its previous result.
    second = run_pipeline(
        first,
        pipeline,
    )

    # Values, data types, row indices, and column order must match.
    assert_frame_equal(first, second)


def test_duplicate_removal_is_idempotent() -> None:
    """Removing duplicate rows twice must not remove additional rows."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            duplicate_step(
                "remove-duplicates",
            )
        ]
    )

    assert_idempotent(
        dataframe,
        pipeline,
    )


def test_constant_imputation_is_idempotent() -> None:
    """Constant replacement must remain stable after the first run."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(
                "fill-age",
                "constant",
                ["age"],
                fill_value=25,
            )
        ]
    )

    assert_idempotent(
        dataframe,
        pipeline,
    )


def test_mean_imputation_is_idempotent() -> None:
    """A second mean-imputation pass must not change completed values."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(
                "fill-age",
                "mean",
                ["age"],
            )
        ]
    )

    assert_idempotent(
        dataframe,
        pipeline,
    )


def test_combined_cleaning_is_idempotent() -> None:
    """Imputation followed by duplicate removal must remain stable."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(
                "fill-age",
                "constant",
                ["age"],
                fill_value=20,
            ),
            duplicate_step(
                "remove-exact-duplicates",
            ),
        ]
    )

    first = run_pipeline(
        dataframe,
        pipeline,
    )

    # Imputation makes the first two customer records identical.
    # The final two records are already exact duplicates.
    assert first["customer_id"].tolist() == [
        101,
        102,
        103,
    ]

    assert_idempotent(
        dataframe,
        pipeline,
    )


def test_drop_rows_is_idempotent() -> None:
    """Rows removed for missing values must stay removed."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(
                "remove-missing-age",
                "drop_rows",
                ["age"],
            )
        ]
    )

    first = run_pipeline(
        dataframe,
        pipeline,
    )

    assert first["age"].isna().sum() == 0

    assert_idempotent(
        dataframe,
        pipeline,
    )


def test_repeated_executions_are_independent() -> None:
    """Running a pipeline twice on the source must produce equal results."""

    dataframe = sample_dataframe()

    pipeline = make_pipeline(
        [
            missing_step(
                "fill-age",
                "constant",
                ["age"],
                fill_value=20,
            ),
            duplicate_step(
                "remove-duplicates",
            ),
        ]
    )

    first = run_pipeline(
        dataframe,
        pipeline,
    )

    second = run_pipeline(
        dataframe,
        pipeline,
    )

    assert_frame_equal(
        first,
        second,
    )


def test_idempotency_checks_preserve_source() -> None:
    """Repeated cleaning must never modify the uploaded dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = make_pipeline(
        [
            missing_step(
                "fill-age",
                "constant",
                ["age"],
                fill_value=20,
            ),
            duplicate_step(
                "remove-duplicates",
            ),
        ]
    )

    assert_idempotent(
        dataframe,
        pipeline,
    )

    assert_frame_equal(
        dataframe,
        original,
    )


def test_non_default_indices_remain_stable() -> None:
    """Repeated cleaning must preserve surviving source-row indices."""

    dataframe = sample_dataframe()

    dataframe.index = [
        10,
        20,
        30,
        40,
        50,
    ]

    pipeline = make_pipeline(
        [
            missing_step(
                "fill-age",
                "constant",
                ["age"],
                fill_value=20,
            ),
            duplicate_step(
                "remove-duplicates",
            ),
        ]
    )

    first = run_pipeline(
        dataframe,
        pipeline,
    )

    assert first.index.tolist() == [
        10,
        30,
        40,
    ]

    assert_idempotent(
        dataframe,
        pipeline,
    )
