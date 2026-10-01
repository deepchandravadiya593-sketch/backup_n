
"""NUMPA v2 — Text cleaning engine.

Supported operations:
    - trim: Remove leading and trailing whitespace.
    - lowercase: Convert text to lowercase.
    - uppercase: Convert text to uppercase.
    - titlecase: Convert text to title case.
    - collapse_spaces: Replace repeated whitespace with one space.
    - replace: Replace literal text or regular-expression matches.
    - standardize_missing: Convert selected text labels to missing values.

All operations preserve the original DataFrame and produce
an audit report. Non-string values remain unchanged.
"""

import re
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator


TextStrategy = Literal[
    "trim",
    "lowercase",
    "uppercase",
    "titlecase",
    "collapse_spaces",
    "replace",
    "standardize_missing",
]


class TextCleaningConfig(BaseModel):
    """User-approved configuration for one text operation."""

    strategy: TextStrategy

    # Explicit selection prevents accidental modification
    # of target, identifier or unrelated columns.
    columns: list[str] = Field(min_length=1)

    # Used by the replace strategy.
    search_text: str | None = None
    replacement_text: str = ""
    use_regex: bool = False
    case_sensitive: bool = True

    # Used by standardize_missing.
    missing_labels: list[str] = Field(
        default_factory=lambda: [
            "",
            "n/a",
            "na",
            "null",
            "none",
            "unknown",
            "missing",
        ]
    )

    @model_validator(mode="after")
    def validate_config(self) -> "TextCleaningConfig":
        """Validate selected columns and replacement settings."""

        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                "Selected columns contain duplicate names."
            )

        if self.strategy == "replace":
            if not self.search_text:
                raise ValueError(
                    "search_text is required for text replacement."
                )

            if self.use_regex:
                try:
                    re.compile(self.search_text)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid regular expression: {exc}"
                    ) from exc

        return self


def _validate_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """Verify that the selected columns exist."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before text cleaning."
        )

    unknown = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if unknown:
        raise ValueError(
            f"Unknown columns: {unknown}"
        )


def _clean_text_value(
    value: Any,
    config: TextCleaningConfig,
) -> Any:
    """Apply the selected operation to one cell."""

    # Preserve missing values and non-string data.
    if not isinstance(value, str):
        return value

    strategy = config.strategy

    if strategy == "trim":
        return value.strip()

    if strategy == "lowercase":
        return value.lower()

    if strategy == "uppercase":
        return value.upper()

    if strategy == "titlecase":
        return value.title()

    if strategy == "collapse_spaces":
        # Collapse repeated spaces, tabs and line breaks.
        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

    if strategy == "replace":
        assert config.search_text is not None

        flags = (
            0
            if config.case_sensitive
            else re.IGNORECASE
        )

        pattern = (
            config.search_text
            if config.use_regex
            else re.escape(config.search_text)
        )

        # For literal replacement, preserve backslashes
        # in the replacement string without interpreting
        # them as regular-expression backreferences.
        replacement = (
            config.replacement_text
            if config.use_regex
            else lambda match: config.replacement_text
        )

        return re.sub(
            pattern,
            replacement,
            value,
            flags=flags,
        )

    if strategy == "standardize_missing":
        # Compare case-insensitively and ignore surrounding
        # whitespace, but leave unrelated values unchanged.
        normalized = value.strip().casefold()

        missing_set = {
            label.strip().casefold()
            for label in config.missing_labels
        }

        if normalized in missing_set:
            return pd.NA

        return value

    raise ValueError(
        f"Unsupported text cleaning strategy: {strategy}"
    )


def clean_text(
    dataframe: pd.DataFrame,
    config: TextCleaningConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a text operation and generate an audit report."""

    _validate_columns(
        dataframe,
        config.columns,
    )

    # Protect the original uploaded dataset.
    cleaned = dataframe.copy(deep=True)

    column_reports: dict[str, dict[str, Any]] = {}
    total_changed = 0
    total_new_missing = 0

    for column in config.columns:
        original = cleaned[column]

        # Object dtype can safely contain mixed string,
        # numeric and missing values.
        transformed = original.astype(
            "object"
        ).map(
            lambda value: _clean_text_value(
                value,
                config,
            )
        )

        # Compare cell-by-cell positionally. This works
        # even when the DataFrame index has duplicates.
        changed_count = 0

        for old, new in zip(
            original.tolist(),
            transformed.tolist(),
        ):
            old_missing = (
                old is None
                or old is pd.NA
                or (
                    isinstance(old, float)
                    and pd.isna(old)
                )
            )

            new_missing = (
                new is None
                or new is pd.NA
                or (
                    isinstance(new, float)
                    and pd.isna(new)
                )
            )

            if old_missing and new_missing:
                continue

            if old_missing != new_missing:
                changed_count += 1
                continue

            if old != new:
                changed_count += 1

        missing_before = int(
            original.isna().sum()
        )

        missing_after = int(
            transformed.isna().sum()
        )

        new_missing = max(
            0,
            missing_after - missing_before,
        )

        cleaned[column] = transformed

        column_reports[column] = {
            "dtype_before": str(original.dtype),
            "dtype_after": str(
                cleaned[column].dtype
            ),
            "changed_values": changed_count,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "new_missing_values": new_missing,
        }

        total_changed += changed_count
        total_new_missing += new_missing

    report: dict[str, Any] = {
        "operation": "text_cleaning",
        "strategy": config.strategy,
        "selected_columns": list(
            config.columns
        ),
        "rows_before": int(
            len(dataframe)
        ),
        "rows_after": int(
            len(cleaned)
        ),
        "columns_before": int(
            len(dataframe.columns)
        ),
        "columns_after": int(
            len(cleaned.columns)
        ),
        "column_reports": column_reports,
        "total_changed_values": total_changed,
        "total_new_missing_values": total_new_missing,
        "status": "completed",
    }

    return cleaned, report
