
"""NUMPA v2 — User-controlled feature selection engine.

Supported strategies:
    keep              Keep explicitly selected columns.
    drop              Remove explicitly selected columns.
    high_missing      Remove columns above a missing-value threshold.
    constant          Remove columns with one or fewer unique values.
    high_correlation  Remove highly correlated numeric features.

Protected columns are never removed automatically.
The original DataFrame is never modified.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


FeatureStrategy = Literal[
    "keep",
    "drop",
    "high_missing",
    "constant",
    "high_correlation",
]


class FeatureSelectionConfig(BaseModel):
    """User-approved feature-selection configuration."""

    strategy: FeatureStrategy

    # Required for manual keep/drop. For automatic strategies,
    # an empty list means all eligible columns.
    columns: list[str] = Field(default_factory=list)

    # Prevent accidental removal of important columns.
    protected_columns: list[str] = Field(default_factory=list)

    # Remove columns with missing percentage strictly
    # greater than this threshold.
    missing_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
    )

    # Remove one feature from a pair when the absolute
    # Pearson correlation exceeds this threshold.
    correlation_threshold: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> "FeatureSelectionConfig":
        """Reject duplicate or missing manual selections."""

        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                "Selected columns contain duplicate names."
            )

        if len(self.protected_columns) != len(
            set(self.protected_columns)
        ):
            raise ValueError(
                "Protected columns contain duplicate names."
            )

        if self.strategy in {"keep", "drop"} and not self.columns:
            raise ValueError(
                "Manual keep/drop requires at least one column."
            )

        return self


def _validate_columns(
    dataframe: pd.DataFrame,
    config: FeatureSelectionConfig,
) -> None:
    """Validate the dataset and referenced column names."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before feature selection."
        )

    referenced = (
        set(config.columns)
        | set(config.protected_columns)
    )

    unknown = sorted(
        column
        for column in referenced
        if column not in dataframe.columns
    )

    if unknown:
        raise ValueError(f"Unknown columns: {unknown}")

    if config.strategy == "drop":
        conflicting = (
            set(config.columns)
            & set(config.protected_columns)
        )

        if conflicting:
            raise ValueError(
                "Cannot manually drop protected columns: "
                f"{sorted(conflicting)}"
            )


def _get_candidates(
    dataframe: pd.DataFrame,
    config: FeatureSelectionConfig,
) -> list[str]:
    """Return eligible columns in their original order."""

    selected = (
        set(config.columns)
        if config.columns
        else set(dataframe.columns)
    )

    protected = set(config.protected_columns)

    return [
        column
        for column in dataframe.columns
        if column in selected and column not in protected
    ]


def _find_correlated_columns(
    dataframe: pd.DataFrame,
    candidates: list[str],
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Find redundant numeric features using Pearson correlation.

    Keep earlier columns and remove later columns when their
    absolute pairwise correlation exceeds the threshold.

    Correlation is calculated only between candidate columns.
    Targets and other protected columns are excluded.
    """

    numeric_columns = [
        column
        for column in candidates
        if (
            pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
            and not pd.api.types.is_bool_dtype(
                dataframe[column]
            )
            and not pd.api.types.is_complex_dtype(
                dataframe[column]
            )
        )
    ]

    if len(numeric_columns) < 2:
        return [], []

    # Treat infinite values as missing for correlation
    # calculation without modifying the original dataset.
    numeric = dataframe[numeric_columns].astype(
        "float64"
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    correlation = numeric.corr(
        method="pearson",
        min_periods=2,
    ).abs()

    removed: list[str] = []
    decisions: list[dict[str, Any]] = []

    # Preserve the first eligible feature in original
    # column order. Compare later features only with
    # features that have not already been removed.
    kept: list[str] = []

    for column in numeric_columns:
        redundant_with: str | None = None
        correlation_value: float | None = None

        for earlier_column in kept:
            value = correlation.loc[
                earlier_column,
                column,
            ]

            if pd.notna(value) and value > threshold:
                redundant_with = earlier_column
                correlation_value = float(value)
                break

        if redundant_with is None:
            kept.append(column)
            continue

        removed.append(column)

        decisions.append({
            "removed_column": column,
            "retained_column": redundant_with,
            "absolute_correlation": correlation_value,
        })

    return removed, decisions


def select_features(
    dataframe: pd.DataFrame,
    config: FeatureSelectionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply feature selection and return an audit report."""

    _validate_columns(dataframe, config)

    # Never modify the original uploaded dataset.
    cleaned = dataframe.copy(deep=True)

    columns_before = list(cleaned.columns)
    protected = set(config.protected_columns)

    removed_columns: list[str] = []
    reasons: dict[str, dict[str, Any]] = {}
    correlation_decisions: list[dict[str, Any]] = []

    if config.strategy == "keep":
        # Always retain protected columns, even if they
        # were omitted from the manual selection.
        keep = set(config.columns) | protected

        removed_columns = [
            column
            for column in columns_before
            if column not in keep
        ]

        for column in removed_columns:
            reasons[column] = {
                "reason": "not_selected",
            }

    elif config.strategy == "drop":
        removed_columns = [
            column
            for column in columns_before
            if column in config.columns
        ]

        for column in removed_columns:
            reasons[column] = {
                "reason": "manual_removal",
            }

    else:
        candidates = _get_candidates(
            dataframe,
            config,
        )

        if config.strategy == "high_missing":
            for column in candidates:
                # An empty dataset has no meaningful
                # missing-value percentage.
                if len(dataframe) == 0:
                    continue

                percentage = float(
                    dataframe[column].isna().mean()
                    * 100
                )

                if percentage > config.missing_threshold:
                    removed_columns.append(column)

                    reasons[column] = {
                        "reason": "high_missing_percentage",
                        "missing_percentage": round(
                            percentage,
                            4,
                        ),
                        "threshold": config.missing_threshold,
                    }

        elif config.strategy == "constant":
            for column in candidates:
                # dropna=False treats missing values as
                # a distinct value. A fully missing column
                # is also considered constant.
                unique_count = int(
                    dataframe[column].nunique(
                        dropna=False
                    )
                )

                if unique_count <= 1:
                    removed_columns.append(column)

                    reasons[column] = {
                        "reason": "constant_feature",
                        "unique_count": unique_count,
                    }

        elif config.strategy == "high_correlation":
            (
                removed_columns,
                correlation_decisions,
            ) = _find_correlated_columns(
                dataframe,
                candidates,
                config.correlation_threshold,
            )

            for decision in correlation_decisions:
                reasons[
                    decision["removed_column"]
                ] = {
                    "reason": "high_correlation",
                    "retained_column": decision[
                        "retained_column"
                    ],
                    "absolute_correlation": decision[
                        "absolute_correlation"
                    ],
                    "threshold": config.correlation_threshold,
                }

    # Perform removal only after all decisions are made.
    cleaned = cleaned.drop(
        columns=removed_columns
    )

    report: dict[str, Any] = {
        "operation": "feature_selection",
        "strategy": config.strategy,
        "selected_columns": list(config.columns),
        "protected_columns": list(
            config.protected_columns
        ),
        "rows_before": int(len(dataframe)),
        "rows_after": int(len(cleaned)),
        "columns_before": len(columns_before),
        "columns_after": int(len(cleaned.columns)),
        "removed_columns": removed_columns,
        "retained_columns": list(cleaned.columns),
        "removed_count": len(removed_columns),
        "removal_reasons": reasons,
        "correlation_decisions": correlation_decisions,
        "status": "completed",
    }

    return cleaned, report
