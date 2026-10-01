
"""NUMPA v2 — Missing-value and duplicate-cleaning integration tests.

Verify that missing-value cleaning and duplicate removal work
correctly when combined in a user-approved cleaning pipeline.

Coverage:
1. Missing-value imputation followed by duplicate removal.
2. Duplicate removal followed by missing-value imputation.
3. Execution order changes the result when appropriate.
4. Selected-column cleaning.
5. Disabled operations.
6. Source-data preservation.
7. Deterministic repeated execution.
8. Empty pipeline behavior.
9. Duplicate removal after constant imputation.
10. Failed configurations do not mutate source data.
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


def sample_dataframe() -> pd.DataFrame:
    """Create data containing missing values and repeated customers."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103, 104],
            "age": [20.0, None, 30.0, 40.0, 50.0],
            "city": [
                "Surat",
                "Rajkot",
                "Rajkot",
                "Ahmedabad",
                "Vadodara",
            ],
        }
    )


def missing_step(
    step_id: str,
    *,
    strategy: str = "constant",
    columns: list[str] | None = None,
    fill_value: object = 30,
    enabled: bool = True,
) -> dict:
    """Create a missing-value step using the pipeline's API contract."""

    parameters = {
        "strategy": strategy,
    }

    if columns is not None:
        parameters["columns"] = columns

    # Constant imputation requires an explicit replacement.
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
    """Create a duplicate-removal step with validated field names."""

    parameters = {
        "strategy": strategy,
    }

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
) -> CleaningPipelineConfig:
    """Validate an ordered cleaning pipeline."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Missing and Duplicate Integration",
            "steps": steps,
            "stop_on_error": True,
        }
    )


def run_pipeline(
    dataframe: pd.DataFrame,
    steps: list[dict],
) -> pd.DataFrame:
    """Execute a pipeline and return its cleaned dataset."""

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        make_pipeline(steps),
    )

    return cleaned


def test_imputation_then_duplicate_removal() -> None:
    """Imputation may create duplicate rows that can then be removed."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101, 102],
            "age": [30.0, None, 40.0],
        }
    )

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-age",
                columns=["age"],
                fill_value=30,
            ),
            duplicate_step(
                "remove-duplicates",
            ),
        ],
    )

    expected = dataframe.iloc[[0, 2]].copy()

    assert_frame_equal(
        cleaned.reset_index(drop=True),
        expected.reset_index(drop=True),
    )

    assert cleaned["age"].tolist() == [30.0, 40.0]


def test_duplicate_removal_then_imputation() -> None:
    """Removing duplicates first must preserve distinct missing rows."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101, 102],
            "age": [30.0, None, 40.0],
        }
    )

    cleaned = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "remove-duplicates",
            ),
            missing_step(
                "fill-age",
                columns=["age"],
                fill_value=30,
            ),
        ],
    )

    # The rows were distinct when duplicate removal ran.
    # Imputation then makes the first two rows identical.
    assert len(cleaned) == 3

    assert cleaned["age"].tolist() == [
        30.0,
        30.0,
        40.0,
    ]


def test_step_order_changes_row_count() -> None:
    """Reversing operations can produce different cleaned datasets."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101, 102],
            "age": [30.0, None, 40.0],
        }
    )

    impute_first = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-age",
                columns=["age"],
            ),
            duplicate_step(
                "deduplicate",
            ),
        ],
    )

    deduplicate_first = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "deduplicate",
            ),
            missing_step(
                "fill-age",
                columns=["age"],
            ),
        ],
    )

    assert len(impute_first) == 2
    assert len(deduplicate_first) == 3


def test_selected_columns_are_respected() -> None:
    """Each operation must use only its approved columns."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101, 102],
            "age": [20.0, None, 40.0],
            "city": ["Surat", None, "Rajkot"],
        }
    )

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-age",
                columns=["age"],
                fill_value=20,
            ),
            duplicate_step(
                "deduplicate-customers",
                columns=["customer_id"],
            ),
        ],
    )

    assert cleaned["customer_id"].tolist() == [101, 102]

    # The unselected city column must retain its source values.
    assert cleaned["city"].tolist() == ["Surat", "Rajkot"]


def test_disabled_imputation_is_skipped() -> None:
    """A disabled imputation step must not fill missing values."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step(
                "disabled-imputation",
                columns=["age"],
                enabled=False,
            ),
            duplicate_step(
                "deduplicate",
                columns=["customer_id"],
            ),
        ],
    )

    assert len(cleaned) == 4

    # The first occurrence of customer 102 has a missing age.
    assert pd.isna(cleaned.loc[1, "age"])


def test_disabled_duplicate_removal_is_skipped() -> None:
    """A disabled duplicate step must preserve repeated customers."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-age",
                columns=["age"],
            ),
            duplicate_step(
                "disabled-deduplication",
                columns=["customer_id"],
                enabled=False,
            ),
        ],
    )

    assert len(cleaned) == len(dataframe)
    assert cleaned["customer_id"].tolist().count(102) == 2


def test_combined_pipeline_preserves_source() -> None:
    """Both operations must leave the uploaded dataset unchanged."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-age",
                columns=["age"],
            ),
            duplicate_step(
                "deduplicate",
                columns=["customer_id"],
            ),
        ],
    )

    assert_frame_equal(dataframe, original)


def test_combined_pipeline_is_deterministic() -> None:
    """Repeated execution must produce the same cleaned dataset."""

    dataframe = sample_dataframe()

    steps = [
        missing_step(
            "fill-age",
            columns=["age"],
        ),
        duplicate_step(
            "deduplicate",
            columns=["customer_id"],
        ),
    ]

    first = run_pipeline(dataframe, steps)
    second = run_pipeline(dataframe, steps)

    assert_frame_equal(first, second)


def test_empty_pipeline_preserves_source() -> None:
    """An empty pipeline must return an unchanged dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    cleaned = run_pipeline(dataframe, [])

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_constant_imputation_can_create_duplicates() -> None:
    """Constant replacement can make previously distinct rows identical."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101],
            "city": ["Surat", None],
        }
    )

    cleaned = run_pipeline(
        dataframe,
        [
            missing_step(
                "fill-city",
                columns=["city"],
                fill_value="Surat",
            ),
            duplicate_step(
                "deduplicate",
            ),
        ],
    )

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["city"] == "Surat"


def test_invalid_missing_strategy_preserves_source() -> None:
    """An invalid strategy must fail without modifying uploaded data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    definition = [
        missing_step(
            "invalid-missing",
            strategy="unsupported_strategy",
            columns=["age"],
        )
    ]

    try:
        pipeline = make_pipeline(definition)
    except ValidationError:
        # Rejecting invalid parameters at model creation is valid.
        assert_frame_equal(dataframe, original)
        return

    # Implementations may instead validate during execution.
    with pytest.raises(ValidationError):
        execute_pipeline(dataframe, pipeline)

    assert_frame_equal(dataframe, original)
