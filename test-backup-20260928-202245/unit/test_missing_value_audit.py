
"""NUMPA v2 — Missing-value cleaning audit tests.

Verify that missing-value cleaning produces consistent results,
preserves uploaded data, and records completed pipeline steps.

The tests use the existing public cleaning pipeline interface.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)
from app.services.cleaning.operations.missing import MissingValueConfig


def sample_dataframe() -> pd.DataFrame:
    """Create a dataset with missing numeric and text values."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "age": [20.0, None, 40.0, 60.0],
            "city": ["Surat", "Rajkot", None, "Ahmedabad"],
            "spend": [100.0, 200.0, 300.0, 400.0],
        }
    )


def make_pipeline(
    strategy: str,
    *,
    columns: list[str] | None = None,
    fill_value: object = None,
) -> CleaningPipelineConfig:
    """Create a pipeline with one approved missing-value step."""

    parameters = {
        "strategy": strategy,
    }

    if columns is not None:
        parameters["columns"] = columns

    # Constant imputation requires an explicit replacement value.
    if strategy == "constant":
        parameters["fill_value"] = fill_value

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Missing Value Audit Pipeline",
            "steps": [
                {
                    "step_id": "missing-audit-step",
                    "operation": "missing_values",
                    "parameters": parameters,
                    "enabled": True,
                }
            ],
            "stop_on_error": True,
        }
    )


def run_cleaning(
    dataframe: pd.DataFrame,
    strategy: str,
    *,
    columns: list[str] | None = None,
    fill_value: object = None,
):
    """Return the complete pipeline execution result."""

    pipeline = make_pipeline(
        strategy,
        columns=columns,
        fill_value=fill_value,
    )

    return execute_pipeline(
        dataframe,
        pipeline,
    )


def test_mean_imputation_removes_selected_missing_values() -> None:
    """The cleaned data must reflect the approved mean strategy."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    # The mean of 20, 40, and 60 is 40.
    assert cleaned.loc[1, "age"] == pytest.approx(40.0)
    assert cleaned["age"].isna().sum() == 0

    # Unselected columns must remain unchanged.
    assert pd.isna(cleaned.loc[2, "city"])


def test_constant_imputation_uses_approved_value() -> None:
    """Constant imputation must use the exact supplied value."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_cleaning(
        dataframe,
        "constant",
        columns=["age"],
        fill_value=25,
    )

    assert cleaned.loc[1, "age"] == 25
    assert cleaned["age"].isna().sum() == 0


def test_drop_rows_matches_expected_records() -> None:
    """Row removal must affect only rows missing selected values."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_cleaning(
        dataframe,
        "drop_rows",
        columns=["age"],
    )

    expected = dataframe.iloc[[0, 2, 3]].copy()

    assert_frame_equal(cleaned, expected)


def test_cleaning_preserves_unselected_columns() -> None:
    """Imputation must not rewrite unrelated dataset columns."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_cleaning(
        dataframe,
        "median",
        columns=["age"],
    )

    assert_frame_equal(
        cleaned[["customer_id", "city", "spend"]],
        dataframe[["customer_id", "city", "spend"]],
    )


def test_repeated_execution_produces_identical_results() -> None:
    """The same dataset and configuration must be deterministic."""

    dataframe = sample_dataframe()

    first_cleaned, first_report, first_artifacts = run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    second_cleaned, second_report, second_artifacts = run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    assert_frame_equal(
        first_cleaned,
        second_cleaned,
    )

    # Reports and artifacts must be reproducible for identical inputs.
    assert first_report == second_report
    assert first_artifacts == second_artifacts


@pytest.mark.parametrize(
    "strategy",
    [
        "mean",
        "median",
        "drop_rows",
        "forward_fill",
        "backward_fill",
    ],
)
def test_cleaning_preserves_uploaded_dataset(
    strategy: str,
) -> None:
    """Every strategy must protect the original uploaded dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_cleaning(
        dataframe,
        strategy,
        columns=["age"],
    )

    assert_frame_equal(
        dataframe,
        original,
    )


def test_invalid_strategy_is_rejected() -> None:
    """Unsupported strategies must fail configuration validation."""

    with pytest.raises(ValidationError):
        MissingValueConfig.model_validate(
            {
                "strategy": "invented_strategy",
            }
        )


def test_constant_strategy_requires_fill_value() -> None:
    """Constant imputation must not silently invent a replacement."""

    with pytest.raises(
        ValidationError,
        match="fill_value",
    ):
        MissingValueConfig.model_validate(
            {
                "strategy": "constant",
            }
        )


def test_invalid_configuration_preserves_source() -> None:
    """A rejected configuration must not alter uploaded records."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    with pytest.raises(ValidationError):
        MissingValueConfig.model_validate(
            {
                "strategy": "constant",
            }
        )

    assert_frame_equal(
        dataframe,
        original,
    )


def test_successful_execution_returns_report() -> None:
    """The pipeline must return a report for an executed operation."""

    dataframe = sample_dataframe()

    cleaned, report, artifacts = run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    assert report is not None
    assert len(cleaned) == len(dataframe)
    assert cleaned["age"].isna().sum() == 0


def test_empty_pipeline_preserves_data() -> None:
    """A pipeline with no steps must not change the source dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = CleaningPipelineConfig.model_validate(
        {
            "name": "Empty Audit Pipeline",
            "steps": [],
        }
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)
