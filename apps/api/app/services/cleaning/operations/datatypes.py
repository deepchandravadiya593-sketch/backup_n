
"""NUMPA v2 — Data type conversion engine.

Supports controlled conversion to:
    - string
    - integer
    - float
    - boolean
    - category
    - datetime

The original DataFrame is never modified.
Failed conversions are reported in the audit log.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


TargetType = Literal[
    "string",
    "integer",
    "float",
    "boolean",
    "category",
    "datetime",
]

ErrorStrategy = Literal["raise", "coerce"]


class DataTypeConfig(BaseModel):
    """User-approved data type conversion configuration."""

    # Map each selected column to its desired data type.
    conversions: dict[str, TargetType] = Field(min_length=1)

    # raise: reject invalid values without changing the dataset.
    # coerce: replace invalid values with missing values.
    errors: ErrorStrategy = "raise"

    # Used for datetime conversion.
    # Specify a format such as "%d/%m/%Y" when dates are ambiguous.
    datetime_format: str | None = None

    # Optional explicit text values for boolean conversion.
    true_values: list[str] = Field(
        default_factory=lambda: ["true", "yes", "1"]
    )
    false_values: list[str] = Field(
        default_factory=lambda: ["false", "no", "0"]
    )

    @model_validator(mode="after")
    def validate_config(self) -> "DataTypeConfig":
        """Reject ambiguous boolean mappings."""

        true_set = {
            value.strip().casefold()
            for value in self.true_values
        }
        false_set = {
            value.strip().casefold()
            for value in self.false_values
        }

        if true_set & false_set:
            raise ValueError(
                "Boolean true and false values must not overlap."
            )

        return self


def _convert_boolean(
    series: pd.Series,
    config: DataTypeConfig,
) -> pd.Series:
    """Convert supported text and numeric values to booleans."""

    true_set = {
        value.strip().casefold()
        for value in config.true_values
    }
    false_set = {
        value.strip().casefold()
        for value in config.false_values
    }

    converted: list[Any] = []

    for value in series:
        if pd.isna(value):
            converted.append(pd.NA)
            continue

        # Normalize text values for case-insensitive matching.
        normalized = str(value).strip().casefold()

        if normalized in true_set:
            converted.append(True)

        elif normalized in false_set:
            converted.append(False)

        elif config.errors == "coerce":
            converted.append(pd.NA)

        else:
            raise ValueError(
                f"Cannot convert value '{value}' to boolean."
            )

    return pd.Series(
        converted,
        index=series.index,
        dtype="boolean",
    )


def _convert_column(
    series: pd.Series,
    target_type: TargetType,
    config: DataTypeConfig,
) -> pd.Series:
    """Convert one column according to the requested data type."""

    if target_type == "string":
        return series.astype("string")

    if target_type == "category":
        return series.astype("category")

    if target_type == "boolean":
        return _convert_boolean(series, config)

    if target_type == "datetime":
        return pd.to_datetime(
            series,
            format=config.datetime_format,
            errors=config.errors,
        )

    if target_type == "float":
        numeric = pd.to_numeric(
            series,
            errors=config.errors,
        )

        # Nullable Float64 preserves missing values.
        return numeric.astype("Float64")

    if target_type == "integer":
        numeric = pd.to_numeric(
            series,
            errors=config.errors,
        )

        # Fractional values must not be silently truncated.
        valid = numeric.dropna()

        fractional = valid[
            np.mod(valid.to_numpy(dtype=float), 1) != 0
        ]

        if not fractional.empty:
            if config.errors == "raise":
                raise ValueError(
                    "Fractional values cannot be converted "
                    "to integers without losing information."
                )

            # Replace fractional values with missing values.
            numeric = numeric.copy()
            numeric.loc[fractional.index] = np.nan

        return numeric.astype("Int64")

    raise ValueError(
        f"Unsupported target data type: {target_type}"
    )


def _count_failed_conversions(
    original: pd.Series,
    converted: pd.Series,
) -> int:
    """Count newly introduced missing values."""

    original_non_null = original.notna()
    converted_null = converted.isna()

    return int(
        (original_non_null & converted_null).sum()
    )


def convert_data_types(
    dataframe: pd.DataFrame,
    config: DataTypeConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert selected column types and generate an audit report."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before conversion."
        )

    unknown_columns = [
        column
        for column in config.conversions
        if column not in dataframe.columns
    ]

    if unknown_columns:
        raise ValueError(
            f"Unknown columns: {unknown_columns}"
        )

    # Work on a deep copy to protect the uploaded dataset.
    cleaned = dataframe.copy(deep=True)

    column_reports: dict[str, dict[str, Any]] = {}

    for column, target_type in config.conversions.items():
        original = cleaned[column]
        original_dtype = str(original.dtype)
        missing_before = int(original.isna().sum())

        try:
            converted = _convert_column(
                original,
                target_type,
                config,
            )

        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(
                f"Failed to convert column '{column}' "
                f"to {target_type}: {exc}"
            ) from exc

        failures = _count_failed_conversions(
            original,
            converted,
        )

        cleaned[column] = converted

        column_reports[column] = {
            "original_dtype": original_dtype,
            "target_type": target_type,
            "result_dtype": str(cleaned[column].dtype),
            "missing_before": missing_before,
            "missing_after": int(
                cleaned[column].isna().sum()
            ),
            "failed_conversions": failures,
        }

    report: dict[str, Any] = {
        "operation": "data_type_conversion",
        "selected_columns": list(
            config.conversions.keys()
        ),
        "errors": config.errors,
        "rows_before": int(len(dataframe)),
        "rows_after": int(len(cleaned)),
        "columns_before": int(len(dataframe.columns)),
        "columns_after": int(len(cleaned.columns)),
        "column_reports": column_reports,
        "total_failed_conversions": sum(
            item["failed_conversions"]
            for item in column_reports.values()
        ),
        "status": "completed",
    }

    return cleaned, report
