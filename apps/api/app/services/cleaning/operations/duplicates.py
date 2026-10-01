
"""NUMPA v2 — Duplicate-row cleaning engine.

Detect and remove duplicate rows using either all columns or a
user-selected subset.

Supported strategies:
- keep_first: Keep the first row in each duplicate group.
- keep_last: Keep the last row in each duplicate group.
- remove_all: Remove every row belonging to a duplicate group.

The original DataFrame is never modified.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Only these duplicate-removal strategies are supported.
DuplicateStrategy = Literal[
    "keep_first",
    "keep_last",
    "remove_all",
]


class DuplicateConfig(BaseModel):
    """User-approved duplicate-removal configuration."""

    # Reject unknown fields such as "keep" or misspelled options.
    model_config = ConfigDict(extra="forbid")

    strategy: DuplicateStrategy = "keep_first"

    # An empty list means that every column is used for matching.
    columns: list[str] = Field(default_factory=list)

    # Optionally ignore surrounding whitespace and capitalization.
    normalize_text: bool = False

    @model_validator(mode="after")
    def validate_columns(self) -> "DuplicateConfig":
        """Reject repeated or empty column names."""

        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                "The selected columns contain duplicate names."
            )

        if any(not column.strip() for column in self.columns):
            raise ValueError(
                "Selected column names cannot be empty."
            )

        return self


def _build_matching_frame(
    dataframe: pd.DataFrame,
    selected_columns: list[str],
    normalize_text: bool,
) -> pd.DataFrame:
    """Build an independent DataFrame for duplicate comparison."""

    # Copy the selected columns to protect the source dataset.
    matching = dataframe.loc[:, selected_columns].copy(deep=True)

    if normalize_text:
        for column in selected_columns:
            # Normalize strings only. Preserve numbers and missing values.
            matching[column] = matching[column].map(
                lambda value: value.strip().casefold()
                if isinstance(value, str)
                else value
            )

    return matching


def _validate_input(
    dataframe: pd.DataFrame,
    config: DuplicateConfig,
) -> list[str]:
    """Validate the input and resolve the matching columns."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not isinstance(config, DuplicateConfig):
        raise TypeError(
            "Expected a validated DuplicateConfig instance."
        )

    # Ambiguous column names make subset selection unsafe.
    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before duplicate detection."
        )

    # This operation requires at least one column.
    if len(dataframe.columns) == 0:
        raise ValueError(
            "At least one column is required for duplicate detection."
        )

    selected_columns = (
        list(config.columns)
        if config.columns
        else list(dataframe.columns)
    )

    # Check selected columns before accessing the DataFrame.
    unknown_columns = [
        column
        for column in selected_columns
        if column not in dataframe.columns
    ]

    if unknown_columns:
        raise ValueError(
            f"Unknown columns: {unknown_columns}"
        )

    return selected_columns


def _duplicate_masks(
    matching: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return masks for extra duplicates and complete duplicate groups."""

    try:
        # Marks every duplicate except the first occurrence.
        extra_duplicate_mask = matching.duplicated(
            keep="first"
        )

        # Marks all rows belonging to a duplicate group.
        all_duplicate_mask = matching.duplicated(
            keep=False
        )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Selected columns contain values that cannot "
            "be compared for duplicate detection."
        ) from exc

    return extra_duplicate_mask, all_duplicate_mask


def detect_duplicates(
    dataframe: pd.DataFrame,
    config: DuplicateConfig,
) -> dict[str, Any]:
    """Analyze duplicate groups without removing rows."""

    selected_columns = _validate_input(
        dataframe,
        config,
    )

    matching = _build_matching_frame(
        dataframe,
        selected_columns,
        config.normalize_text,
    )

    extra_mask, all_mask = _duplicate_masks(matching)

    total_rows = int(len(dataframe))
    extra_duplicate_rows = int(extra_mask.sum())

    return {
        "total_rows": total_rows,
        "selected_columns": selected_columns,
        "normalize_text": config.normalize_text,
        "extra_duplicate_rows": extra_duplicate_rows,
        "rows_in_duplicate_groups": int(all_mask.sum()),
        "unique_rows": total_rows - extra_duplicate_rows,
    }


def clean_duplicates(
    dataframe: pd.DataFrame,
    config: DuplicateConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove duplicates and return a cleaned copy and audit report."""

    # Validate the source and collect its initial duplicate statistics.
    detection = detect_duplicates(
        dataframe,
        config,
    )

    selected_columns = detection["selected_columns"]

    # Always work on a separate DataFrame.
    cleaned = dataframe.copy(deep=True)

    matching = _build_matching_frame(
        cleaned,
        selected_columns,
        config.normalize_text,
    )

    try:
        if config.strategy == "keep_first":
            # Preserve the earliest row in each duplicate group.
            remove_mask = matching.duplicated(
                keep="first"
            )

        elif config.strategy == "keep_last":
            # Preserve the latest row in each duplicate group.
            remove_mask = matching.duplicated(
                keep="last"
            )

        elif config.strategy == "remove_all":
            # Remove every row in every duplicate group.
            remove_mask = matching.duplicated(
                keep=False
            )

        else:
            # Defensive check in case validation was bypassed.
            raise ValueError(
                f"Unsupported duplicate strategy: {config.strategy}"
            )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Duplicate removal failed while comparing "
            "the selected columns."
        ) from exc

    rows_before = int(len(cleaned))

    # Use positional selection because source indices may be repeated.
    keep_positions = (~remove_mask).to_numpy()
    cleaned = cleaned.iloc[keep_positions].copy()

    rows_after = int(len(cleaned))

    # Report row positions, not index labels, for an unambiguous audit.
    removed_positions = [
        int(position)
        for position, should_remove in enumerate(
            remove_mask.to_numpy()
        )
        if should_remove
    ]

    report: dict[str, Any] = {
        "operation": "remove_duplicates",
        "strategy": config.strategy,
        "selected_columns": selected_columns,
        "normalize_text": config.normalize_text,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "extra_duplicate_rows_before": detection[
            "extra_duplicate_rows"
        ],
        "rows_in_duplicate_groups_before": detection[
            "rows_in_duplicate_groups"
        ],
        "removed_row_positions": removed_positions,
        "status": "completed",
    }

    return cleaned, report
