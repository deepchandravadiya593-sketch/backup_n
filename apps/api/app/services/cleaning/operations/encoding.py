
"""NUMPA v2 — Categorical encoding engine.

Supports one-hot and ordinal encoding with reusable category
mappings. Never modifies the original DataFrame.

For ML workflows, fit on training data and reuse the saved
mappings on validation and test data to prevent data leakage.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


EncodingStrategy = Literal["one_hot", "ordinal"]
UnknownStrategy = Literal["error", "ignore"]


class EncodingConfig(BaseModel):
    """User-approved categorical encoding configuration."""

    strategy: EncodingStrategy

    # Columns to encode. Explicit selection prevents
    # accidental encoding of targets or identifier columns.
    columns: list[str] = Field(min_length=1)

    # What to do when transforming data with unseen categories.
    handle_unknown: UnknownStrategy = "error"

    # One-hot encoding only: optionally remove the first category.
    # Use with care: the all-zero representation may become
    # ambiguous when unknown categories are ignored.
    drop_first: bool = False

    # Ordinal encoding only: explicit category ordering.
    # If omitted, categories are sorted deterministically.
    category_orders: dict[str, list[Any]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> "EncodingConfig":
        """Validate selected columns and category ordering."""

        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Selected columns contain duplicates.")

        unknown_order_columns = (
            set(self.category_orders) - set(self.columns)
        )

        if unknown_order_columns:
            raise ValueError(
                "Category orders reference unselected columns: "
                f"{sorted(unknown_order_columns)}"
            )

        if self.strategy == "one_hot" and self.category_orders:
            raise ValueError(
                "Explicit category orders are supported only "
                "for ordinal encoding."
            )

        return self


def _validate_dataframe(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """Validate the input dataset and selected columns."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before encoding."
        )

    missing_columns = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unknown columns: {missing_columns}"
        )

    for column in columns:
        series = dataframe[column]

        # This initial engine supports text, categorical
        # and boolean features. Numeric category handling
        # will be added through explicit type conversion.
        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            raise ValueError(
                f"Column '{column}' is numeric. "
                "Select a categorical column instead."
            )

        # Reject complex objects that cannot be mapped safely.
        for value in series.dropna():
            if not isinstance(
                value,
                (str, bool, np.bool_, int, float, np.integer, np.floating),
            ):
                raise ValueError(
                    f"Column '{column}' contains an unsupported "
                    "category value."
                )


def _get_categories(
    series: pd.Series,
    explicit_order: list[Any] | None = None,
) -> list[Any]:
    """Return a deterministic list of non-null categories."""

    observed = series.dropna().unique().tolist()

    if explicit_order is not None:
        if len(explicit_order) != len(set(map(str, explicit_order))):
            raise ValueError(
                "Explicit category order contains duplicates."
            )

        # Ensure the supplied order covers every observed value.
        missing = [
            value
            for value in observed
            if value not in explicit_order
        ]

        if missing:
            raise ValueError(
                f"Category order omits observed values: {missing}"
            )

        return list(explicit_order)

    # Sorting by type and representation avoids relying on
    # the order in which rows happen to appear.
    return sorted(
        observed,
        key=lambda value: (
            type(value).__name__,
            str(value),
        ),
    )


def fit_encoding(
    dataframe: pd.DataFrame,
    config: EncodingConfig,
) -> dict[str, Any]:
    """Learn reusable category mappings from a dataset."""

    _validate_dataframe(dataframe, config.columns)

    categories: dict[str, list[Any]] = {}

    for column in config.columns:
        explicit_order = config.category_orders.get(column)

        categories[column] = _get_categories(
            dataframe[column],
            explicit_order=explicit_order,
        )

    return {
        "strategy": config.strategy,
        "columns": list(config.columns),
        "categories": categories,
        "handle_unknown": config.handle_unknown,
        "drop_first": config.drop_first,
        "fitted_rows": int(len(dataframe)),
    }


def transform_encoding(
    dataframe: pd.DataFrame,
    fitted_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply previously learned category mappings."""

    columns = fitted_config["columns"]
    categories = fitted_config["categories"]
    strategy = fitted_config["strategy"]
    handle_unknown = fitted_config["handle_unknown"]
    drop_first = fitted_config["drop_first"]

    _validate_dataframe(dataframe, columns)

    # Preserve the original dataset.
    cleaned = dataframe.copy(deep=True)

    rows_before, columns_before = cleaned.shape
    created_columns: list[str] = []
    unknown_counts: dict[str, int] = {}

    # Reserve every original column name to prevent
    # generated feature names from overwriting user data.
    reserved_names = set(cleaned.columns)

    if strategy == "one_hot":
        generated_frames: list[pd.DataFrame] = []

        for column in columns:
            known_categories = categories[column]
            series = cleaned[column]

            unknown_mask = (
                series.notna()
                & ~series.isin(known_categories)
            )

            unknown_count = int(unknown_mask.sum())
            unknown_counts[column] = unknown_count

            if unknown_count and handle_unknown == "error":
                raise ValueError(
                    f"Column '{column}' contains "
                    f"{unknown_count} unknown category values."
                )

            # When drop_first is enabled, omit the first
            # category from the generated features.
            active_categories = (
                known_categories[1:]
                if drop_first
                else known_categories
            )

            encoded = pd.DataFrame(
                index=cleaned.index,
            )

            for index, category in enumerate(
                active_categories
            ):
                # Use a stable positional suffix so unusual
                # category values cannot break column names.
                new_column = (
                    f"{column}__onehot_{index}"
                )

                if new_column in reserved_names:
                    raise ValueError(
                        "Generated column name conflicts with "
                        f"an existing column: {new_column}"
                    )

                reserved_names.add(new_column)

                # Missing and ignored unknown values become
                # all-zero rows in the generated features.
                encoded[new_column] = (
                    series.eq(category)
                    .fillna(False)
                    .astype("int8")
                )

                created_columns.append(new_column)

            generated_frames.append(encoded)

        # Remove original encoded columns, then append
        # the new indicator columns in deterministic order.
        cleaned = cleaned.drop(columns=columns)

        if generated_frames:
            cleaned = pd.concat(
                [cleaned, *generated_frames],
                axis=1,
            )

    elif strategy == "ordinal":
        for column in columns:
            known_categories = categories[column]
            series = cleaned[column]

            unknown_mask = (
                series.notna()
                & ~series.isin(known_categories)
            )

            unknown_count = int(unknown_mask.sum())
            unknown_counts[column] = unknown_count

            if unknown_count and handle_unknown == "error":
                raise ValueError(
                    f"Column '{column}' contains "
                    f"{unknown_count} unknown category values."
                )

            # Categories receive integer values starting at 0.
            # Missing or ignored unknown values receive -1.
            mapping = {
                category: index
                for index, category in enumerate(
                    known_categories
                )
            }

            cleaned[column] = (
                series.map(mapping)
                .fillna(-1)
                .astype("int64")
            )

    else:
        raise ValueError(
            f"Unsupported encoding strategy: {strategy}"
        )

    report: dict[str, Any] = {
        "operation": "categorical_encoding",
        "strategy": strategy,
        "selected_columns": columns,
        "rows_before": rows_before,
        "rows_after": int(len(cleaned)),
        "columns_before": columns_before,
        "columns_after": int(len(cleaned.columns)),
        "created_columns": created_columns,
        "category_mappings": categories,
        "unknown_category_counts": unknown_counts,
        "status": "completed",
    }

    return cleaned, report


def encode_categories(
    dataframe: pd.DataFrame,
    config: EncodingConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit and apply an encoder in one operation.

    Returns the transformed dataset, audit report and
    fitted configuration for later reuse.
    """

    fitted_config = fit_encoding(
        dataframe,
        config,
    )

    cleaned, report = transform_encoding(
        dataframe,
        fitted_config,
    )

    return cleaned, report, fitted_config
