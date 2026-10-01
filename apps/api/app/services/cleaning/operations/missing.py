
"""NUMPA v2 — Missing-value cleaning operations.

Supports dropping and imputing missing values while preserving
the original DataFrame and producing an auditable operation report.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


# Supported missing-value strategies.
MissingStrategy = Literal[
    "drop_rows",
    "drop_columns",
    "mean",
    "median",
    "mode",
    "constant",
    "forward_fill",
    "backward_fill",
]


class MissingValueConfig(BaseModel):
    """User-approved configuration for a missing-value operation."""

    strategy: MissingStrategy

    # An empty list means all columns.
    columns: list[str] = Field(default_factory=list)

    # Used only by the constant strategy.
    fill_value: Any = None

    # Maximum missing-value percentage for dropping columns.
    # Example: 50 means drop selected columns with >50% missing.
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> "MissingValueConfig":
        """Reject incomplete or contradictory configurations."""

        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError(
                "A fill_value is required for constant imputation."
            )

        return self


def clean_missing_values(
    dataframe: pd.DataFrame,
    config: MissingValueConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one missing-value operation to a DataFrame copy."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before cleaning."
        )

    # Validate explicitly selected columns.
    selected_columns = (
        list(config.columns)
        if config.columns
        else list(dataframe.columns)
    )

    if len(selected_columns) != len(set(selected_columns)):
        raise ValueError("Selected columns contain duplicates.")

    unknown_columns = [
        column
        for column in selected_columns
        if column not in dataframe.columns
    ]

    if unknown_columns:
        raise ValueError(
            f"Unknown columns: {unknown_columns}"
        )

    # Work on a copy to protect the uploaded original.
    cleaned = dataframe.copy(deep=True)

    rows_before, columns_before = cleaned.shape
    missing_before = int(cleaned.isna().sum().sum())

    # Record the number of missing values in selected columns.
    selected_missing_before = int(
        cleaned[selected_columns].isna().sum().sum()
    )

    strategy = config.strategy

    if strategy == "drop_rows":
        # Remove rows containing missing values in any
        # of the selected columns.
        cleaned = cleaned.dropna(
            axis=0,
            subset=selected_columns,
            how="any",
        )

    elif strategy == "drop_columns":
        # Remove selected columns whose missing percentage
        # exceeds the user-defined threshold.
        columns_to_drop = [
            column
            for column in selected_columns
            if (
                cleaned[column].isna().mean() * 100
                > config.threshold
            )
        ]

        cleaned = cleaned.drop(
            columns=columns_to_drop,
        )

    elif strategy in {"mean", "median"}:
        # Mean and median are valid only for numeric columns.
        for column in selected_columns:
            series = cleaned[column]

            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(
                    f"Column '{column}' is not numeric."
                )

            if pd.api.types.is_bool_dtype(series):
                raise ValueError(
                    f"Column '{column}' is boolean. "
                    "Use mode or constant imputation."
                )

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            if numeric.dropna().empty:
                raise ValueError(
                    f"Column '{column}' has no finite numeric "
                    "values available for imputation."
                )

            if strategy == "mean":
                replacement = numeric.mean()
            else:
                replacement = numeric.median()

            # Replace only originally missing values.
            cleaned[column] = series.fillna(replacement)

    elif strategy == "mode":
        # Fill missing values with the most frequent
        # non-null value in each selected column.
        for column in selected_columns:
            modes = cleaned[column].mode(
                dropna=True,
            )

            if modes.empty:
                raise ValueError(
                    f"Column '{column}' has no available mode."
                )

            cleaned[column] = cleaned[column].fillna(
                modes.iloc[0]
            )

    elif strategy == "constant":
        # Fill missing values with the user-provided value.
        # Pandas will raise if the value is incompatible
        # with the column's dtype.
        for column in selected_columns:
            cleaned[column] = cleaned[column].fillna(
                config.fill_value
            )

    elif strategy == "forward_fill":
        # Carry the previous non-null value forward.
        # Row order matters for this operation.
        cleaned[selected_columns] = (
            cleaned[selected_columns].ffill()
        )

    elif strategy == "backward_fill":
        # Use the next non-null value.
        # Row order matters for this operation.
        cleaned[selected_columns] = (
            cleaned[selected_columns].bfill()
        )

    # Calculate the results after cleaning.
    rows_after, columns_after = cleaned.shape
    missing_after = int(cleaned.isna().sum().sum())

    remaining_selected_columns = [
        column
        for column in selected_columns
        if column in cleaned.columns
    ]

    selected_missing_after = int(
        cleaned[remaining_selected_columns]
        .isna()
        .sum()
        .sum()
    )

    # Keep the original row index so changes can be audited.
    report: dict[str, Any] = {
        "operation": "missing_values",
        "strategy": strategy,
        "selected_columns": selected_columns,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "columns_before": columns_before,
        "columns_after": columns_after,
        "columns_removed": columns_before - columns_after,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "selected_missing_before": selected_missing_before,
        "selected_missing_after": selected_missing_after,
        "status": "completed",
    }

    return cleaned, report
