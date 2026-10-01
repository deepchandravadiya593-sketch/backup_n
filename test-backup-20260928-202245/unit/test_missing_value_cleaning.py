
"""NUMPA v2 — Missing-value cleaning tests.

Test missing-value cleaning through the existing pipeline API.

Coverage:
1. Default configuration validation.
2. Supported strategy validation.
3. Unsupported strategy rejection.
4. Invalid threshold rejection.
5. Unknown pipeline operation rejection.
6. Empty pipeline preservation.
7. Disabled missing-value step.
8. Mean imputation.
9. Median imputation.
10. Row removal.
11. Deterministic execution.
12. Source dataset protection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.services.cleaning.operations.missing import (
    MissingValueConfig,
)

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


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
    enabled: bool = True,
) -> CleaningPipelineConfig:
    """Build a pipeline containing one missing-value operation."""

    parameters = {
        "strategy": strategy,
    }

    if columns is not None:
        parameters["columns"] = columns

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Missing Value Test Pipeline",
            "steps": [
                {
                    "step_id": "handle-missing",
                    "operation": "missing_values",
                    "parameters": parameters,
                    "enabled": enabled,
                }
            ],
        }
    )


def run_cleaning(
    dataframe: pd.DataFrame,
    strategy: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Execute one approved missing-value cleaning operation."""

    pipeline = make_pipeline(
        strategy,
        columns=columns,
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    return cleaned


def test_strategy_is_required() -> None:
    """Missing-value cleaning requires an explicit user-approved strategy."""

    # A strategy must not be selected silently.
    with pytest.raises(ValidationError):
        MissingValueConfig()

    # An explicitly selected strategy must be accepted.
    config = MissingValueConfig(strategy="drop_rows")

    assert config.strategy == "drop_rows"

@pytest.mark.parametrize(
    "strategy",
    [
        "drop_rows",
        "drop_columns",
        "mean",
        "median",
        "mode",
        "constant",
        "forward_fill",
        "backward_fill",
    ],
)
def test_supported_strategies_are_valid(
    strategy: str,
) -> None:
    """All eight supported strategies must accept valid configurations."""

    # Constant imputation requires an explicit replacement value.
    parameters = {
        "strategy": strategy,
    }

    if strategy == "constant":
        parameters["fill_value"] = 0

    config = MissingValueConfig.model_validate(parameters)

    assert config.strategy == strategy

    if strategy == "constant":
        assert config.fill_value == 0


def test_unsupported_strategy_is_rejected() -> None:
    """Unknown strategies must not be silently accepted."""

    with pytest.raises(ValidationError):
        MissingValueConfig.model_validate(
            {
                "strategy": "unsupported_strategy",
            }
        )


@pytest.mark.parametrize(
    "threshold",
    [-1, 101],
)
def test_invalid_threshold_is_rejected(
    threshold: int,
) -> None:
    """The missing-value threshold must remain between 0 and 100."""

    with pytest.raises(ValidationError):
        MissingValueConfig.model_validate(
            {
                "strategy": "drop_columns",
                "threshold": threshold,
            }
        )


def test_unknown_pipeline_operation_is_rejected() -> None:
    """A misspelled operation name must fail pipeline validation."""

    with pytest.raises(ValidationError):
        CleaningPipelineConfig.model_validate(
            {
                "steps": [
                    {
                        "step_id": "invalid",
                        "operation": "missing",
                        "parameters": {
                            "strategy": "mean",
                        },
                    }
                ]
            }
        )


def test_empty_pipeline_preserves_dataset() -> None:
    """An empty pipeline must leave every value unchanged."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = CleaningPipelineConfig.model_validate(
        {
            "name": "Empty Pipeline",
            "steps": [],
        }
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_disabled_missing_step_preserves_dataset() -> None:
    """A disabled operation must not impute or remove any values."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    pipeline = make_pipeline(
        "drop_rows",
        enabled=False,
    )

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_mean_imputation() -> None:
    """Mean imputation must replace the missing age with 40."""

    dataframe = sample_dataframe()

    cleaned = run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    assert cleaned["age"].isna().sum() == 0
    assert cleaned.loc[1, "age"] == pytest.approx(40.0)

    # An unrelated missing text value must remain unchanged.
    assert pd.isna(cleaned.loc[2, "city"])


def test_median_imputation() -> None:
    """Median imputation must replace the missing age with 40."""

    dataframe = sample_dataframe()

    cleaned = run_cleaning(
        dataframe,
        "median",
        columns=["age"],
    )

    assert cleaned["age"].isna().sum() == 0
    assert cleaned.loc[1, "age"] == pytest.approx(40.0)


def test_drop_rows_with_selected_column() -> None:
    """Only rows missing values in the selected column should be removed."""

    dataframe = sample_dataframe()

    cleaned = run_cleaning(
        dataframe,
        "drop_rows",
        columns=["age"],
    )

    expected = dataframe.iloc[
        [0, 2, 3]
    ].copy()

    assert_frame_equal(cleaned, expected)


def test_mean_imputation_is_deterministic() -> None:
    """Identical inputs and parameters must produce identical results."""

    dataframe = sample_dataframe()

    first = run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    second = run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    assert_frame_equal(first, second)


def test_missing_value_cleaning_preserves_source() -> None:
    """Cleaning must never modify the original uploaded dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_cleaning(
        dataframe,
        "mean",
        columns=["age"],
    )

    assert_frame_equal(dataframe, original)
