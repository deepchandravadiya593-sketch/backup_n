
"""NUMPA v2 — Missing-value cleaning edge-case tests.

Test difficult missing-value scenarios through the existing
cleaning pipeline.

Coverage:
- Constant imputation.
- Missing fill-value validation.
- Forward and backward filling.
- Consecutive missing values.
- Entirely missing columns.
- Non-default row indices.
- Empty datasets.
- Source-data preservation.
- Repeated execution.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.services.cleaning.operations.missing import MissingValueConfig
from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


def run_missing_cleaning(
    dataframe: pd.DataFrame,
    strategy: str,
    *,
    columns: list[str] | None = None,
    fill_value: object = None,
) -> pd.DataFrame:
    """Run a missing-value operation using valid pipeline parameters."""

    # Every operation must have an explicit strategy.
    parameters = {
        "strategy": strategy,
    }

    if columns is not None:
        parameters["columns"] = columns

    # Only constant imputation requires a replacement value.
    if strategy == "constant":
        parameters["fill_value"] = fill_value

    pipeline = CleaningPipelineConfig.model_validate(
        {
            "name": "Missing Value Edge-Case Pipeline",
            "steps": [
                {
                    "step_id": "handle-missing-values",
                    "operation": "missing_values",
                    "parameters": parameters,
                    "enabled": True,
                }
            ],
        }
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    return cleaned


def test_constant_imputation_replaces_missing_values() -> None:
    """An approved constant must replace missing numeric values."""

    dataframe = pd.DataFrame(
        {
            "age": [20.0, None, 40.0],
            "city": ["Surat", "Rajkot", None],
        }
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "constant",
        columns=["age"],
        fill_value=0,
    )

    assert cleaned["age"].tolist() == [20.0, 0.0, 40.0]

    # Columns outside the approved selection must remain unchanged.
    assert pd.isna(cleaned.loc[2, "city"])


def test_constant_imputation_requires_fill_value() -> None:
    """Constant imputation must not invent a replacement value."""

    with pytest.raises(
        ValidationError,
        match="fill_value",
    ):
        MissingValueConfig.model_validate(
            {
                "strategy": "constant",
            }
        )


def test_forward_fill_handles_consecutive_gaps() -> None:
    """Forward filling must propagate the previous available value."""

    dataframe = pd.DataFrame(
        {
            "reading": [10.0, None, None, 40.0],
        }
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "forward_fill",
        columns=["reading"],
    )

    assert cleaned["reading"].tolist() == [
        10.0,
        10.0,
        10.0,
        40.0,
    ]


def test_backward_fill_handles_consecutive_gaps() -> None:
    """Backward filling must use the next available value."""

    dataframe = pd.DataFrame(
        {
            "reading": [10.0, None, None, 40.0],
        }
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "backward_fill",
        columns=["reading"],
    )

    assert cleaned["reading"].tolist() == [
        10.0,
        40.0,
        40.0,
        40.0,
    ]


def test_forward_fill_preserves_leading_missing_value() -> None:
    """Forward filling cannot fill a gap without a previous value."""

    dataframe = pd.DataFrame(
        {
            "reading": [None, 20.0, None],
        }
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "forward_fill",
        columns=["reading"],
    )

    assert pd.isna(cleaned.iloc[0]["reading"])
    assert cleaned.iloc[1]["reading"] == 20.0
    assert cleaned.iloc[2]["reading"] == 20.0


def test_backward_fill_preserves_trailing_missing_value() -> None:
    """Backward filling cannot fill a gap without a later value."""

    dataframe = pd.DataFrame(
        {
            "reading": [None, 20.0, None],
        }
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "backward_fill",
        columns=["reading"],
    )

    assert cleaned.iloc[0]["reading"] == 20.0
    assert cleaned.iloc[1]["reading"] == 20.0
    assert pd.isna(cleaned.iloc[2]["reading"])


def test_drop_rows_with_entirely_missing_column() -> None:
    """Removing rows with missing values may produce an empty dataset."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "unknown_value": [None, None, None],
        }
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "drop_rows",
        columns=["unknown_value"],
    )

    assert cleaned.empty
    assert cleaned.columns.tolist() == dataframe.columns.tolist()


def test_drop_rows_preserves_original_indices() -> None:
    """Surviving rows must retain their original index labels."""

    dataframe = pd.DataFrame(
        {
            "age": [20.0, None, 40.0],
        },
        index=[10, 20, 30],
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "drop_rows",
        columns=["age"],
    )

    assert cleaned.index.tolist() == [10, 30]
    assert cleaned["age"].tolist() == [20.0, 40.0]


def test_empty_dataset_remains_empty() -> None:
    """A row-removal operation must safely handle an empty dataset."""

    dataframe = pd.DataFrame(
        {
            "age": pd.Series(dtype="float64"),
        }
    )

    cleaned = run_missing_cleaning(
        dataframe,
        "drop_rows",
        columns=["age"],
    )

    assert_frame_equal(cleaned, dataframe)


def test_constant_imputation_preserves_source() -> None:
    """Replacing missing values must not mutate the uploaded dataset."""

    dataframe = pd.DataFrame(
        {
            "age": [20.0, None, 40.0],
        }
    )

    original = dataframe.copy(deep=True)

    run_missing_cleaning(
        dataframe,
        "constant",
        columns=["age"],
        fill_value=0,
    )

    assert_frame_equal(dataframe, original)


def test_repeated_constant_imputation_is_deterministic() -> None:
    """The same configuration must produce the same cleaned dataset."""

    dataframe = pd.DataFrame(
        {
            "age": [20.0, None, 40.0],
        }
    )

    first = run_missing_cleaning(
        dataframe,
        "constant",
        columns=["age"],
        fill_value=0,
    )

    second = run_missing_cleaning(
        dataframe,
        "constant",
        columns=["age"],
        fill_value=0,
    )

    assert_frame_equal(first, second)
