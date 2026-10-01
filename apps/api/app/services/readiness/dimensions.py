
"""NUMPA v2 - Deterministic ML readiness scoring dimensions.

Calculates preliminary readiness measurements for the seven
default dimensions.

This module does not calculate an overall readiness score.
Unimplemented checks are reported as pending rather than
receiving an artificial passing score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas.readiness import (
    DEFAULT_DIMENSION_WEIGHTS,
    MLTask,
    ReadinessDimensionResult,
    ReadinessIssue,
)


# Initial, documented thresholds. Later, the configuration
# service will supply approved task-specific overrides.
MIN_ROWS = 100
MIN_CLASS_COUNT = 10
HIGH_CARDINALITY_RATIO = 0.90
MAX_MISSING_PERCENTAGE = 50.0


def _percentage(part: int, total: int) -> float:
    """Calculate a bounded percentage safely."""

    if total <= 0:
        return 0.0

    return round(100.0 * part / total, 4)


def _score_from_penalty(penalty: float) -> float:
    """Convert a penalty into a score between zero and 100."""

    return round(max(0.0, min(100.0, 100.0 - penalty)), 2)


def _issue(
    code: str,
    severity: str,
    dimension: str,
    message: str,
    *,
    column: str | None = None,
    recommendation: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> ReadinessIssue:
    """Create a consistent, explainable scoring issue."""

    return ReadinessIssue(
        code=code,
        severity=severity,
        dimension=dimension,
        message=message,
        column=column,
        recommendation=recommendation,
        evidence=evidence or {},
    )


def _result(
    name: str,
    score: float,
    explanation: str,
    *,
    issues: list[ReadinessIssue] | None = None,
    metrics: dict[str, Any] | None = None,
) -> ReadinessDimensionResult:
    """Build a dimension result using the default weight."""

    weight = DEFAULT_DIMENSION_WEIGHTS[name]

    if score >= 80:
        status = "ready"
    elif score >= 50:
        status = "needs_attention"
    else:
        status = "not_ready"

    return ReadinessDimensionResult(
        name=name,
        score=round(score, 2),
        weight=weight,
        weighted_score=round(score * weight / 100.0, 4),
        status=status,
        explanation=explanation,
        issues=issues or [],
        metrics=metrics or {},
    )


def score_completeness(
    dataframe: pd.DataFrame,
) -> ReadinessDimensionResult:
    """Score the proportion of non-missing dataset cells."""

    name = "completeness"
    total_cells = int(dataframe.size)
    missing_cells = int(dataframe.isna().sum().sum())

    missing_percentage = _percentage(
        missing_cells,
        total_cells,
    )

    # Every missing cell contributes proportionally.
    score = _score_from_penalty(missing_percentage)
    issues = []

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())

        if missing_count == 0:
            continue

        column_percentage = _percentage(
            missing_count,
            len(dataframe),
        )

        issues.append(
            _issue(
                "MISSING_VALUES",
                (
                    "high"
                    if column_percentage > MAX_MISSING_PERCENTAGE
                    else "medium"
                ),
                name,
                "The column contains missing values.",
                column=str(column),
                recommendation=(
                    "Inspect the missing-value pattern before "
                    "choosing an imputation or removal strategy."
                ),
                evidence={
                    "missing_count": missing_count,
                    "missing_percentage": column_percentage,
                },
            )
        )

    return _result(
        name,
        score,
        "Completeness is based on the percentage of non-missing cells.",
        issues=issues,
        metrics={
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": missing_percentage,
        },
    )


def score_validity_consistency(
    dataframe: pd.DataFrame,
) -> ReadinessDimensionResult:
    """Check non-finite numeric values and inconsistent column names.

    Domain-specific rules such as valid date ranges or permitted
    business values are not assessed in this preliminary version.
    """

    name = "validity_consistency"
    issues = []

    numeric = dataframe.select_dtypes(include="number")
    numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)

    # Count infinity separately from ordinary missing values.
    infinite_count = int(np.isinf(numeric_values).sum())

    numeric_cells = int(numeric_values.size)
    infinite_percentage = _percentage(
        infinite_count,
        numeric_cells,
    )

    # Detect column names that differ only in surrounding spaces.
    column_names = [str(column) for column in dataframe.columns]
    normalized_names = [
        column.strip().casefold()
        for column in column_names
    ]

    normalized_duplicate_count = (
        len(normalized_names) - len(set(normalized_names))
    )

    if infinite_count:
        issues.append(
            _issue(
                "INFINITE_NUMERIC_VALUES",
                "high",
                name,
                "Numeric columns contain infinite values.",
                recommendation=(
                    "Review infinite values before training."
                ),
                evidence={"count": infinite_count},
            )
        )

    if normalized_duplicate_count:
        issues.append(
            _issue(
                "INCONSISTENT_COLUMN_NAMES",
                "medium",
                name,
                "Column names collide after whitespace and case normalization.",
                recommendation=(
                    "Rename columns so each normalized name is unique."
                ),
                evidence={
                    "collision_count": normalized_duplicate_count,
                },
            )
        )

    penalty = min(
        100.0,
        infinite_percentage
        + (20.0 if normalized_duplicate_count else 0.0),
    )

    return _result(
        name,
        _score_from_penalty(penalty),
        (
            "Preliminary validity checks cover infinite numeric "
            "values and normalized column-name collisions. "
            "Business-rule validation is not yet included."
        ),
        issues=issues,
        metrics={
            "infinite_numeric_values": infinite_count,
            "infinite_percentage": infinite_percentage,
            "normalized_name_collisions": normalized_duplicate_count,
        },
    )


def score_record_integrity(
    dataframe: pd.DataFrame,
) -> ReadinessDimensionResult:
    """Measure exact duplicate rows when pandas can compare them."""

    name = "record_integrity"
    issues = []

    try:
        duplicate_count = int(
            dataframe.duplicated(keep="first").sum()
        )
    except (TypeError, ValueError):
        # Nested objects can make exact duplicate detection
        # impossible. Do not claim a passing score.
        raise ValueError(
            "Record integrity could not be scored because "
            "some column values are not comparable."
        )

    duplicate_percentage = _percentage(
        duplicate_count,
        len(dataframe),
    )

    if duplicate_count:
        issues.append(
            _issue(
                "DUPLICATE_ROWS",
                "medium",
                name,
                "The dataset contains exact duplicate rows.",
                recommendation=(
                    "Determine whether duplicates are genuine "
                    "repeated observations before removing them."
                ),
                evidence={
                    "duplicate_rows": duplicate_count,
                    "duplicate_percentage": duplicate_percentage,
                },
            )
        )

    return _result(
        name,
        _score_from_penalty(duplicate_percentage),
        "Record integrity is based on exact duplicate-row frequency.",
        issues=issues,
        metrics={
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": duplicate_percentage,
        },
    )


def score_feature_suitability(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    excluded_columns: list[str] | None = None,
) -> ReadinessDimensionResult:
    """Detect empty, constant and high-missing feature columns."""

    name = "feature_suitability"
    excluded = set(excluded_columns or [])

    # Never treat the supervised target as an input feature.
    if target_column:
        excluded.add(target_column)

    features = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]

    issues = []

    if not features:
        issues.append(
            _issue(
                "NO_FEATURES",
                "critical",
                name,
                "No input feature columns remain.",
                recommendation="Select at least one suitable feature.",
            )
        )

        return _result(
            name,
            0.0,
            "No input features are available.",
            issues=issues,
            metrics={"feature_count": 0},
        )

    problematic_columns = set()

    for column in features:
        series = dataframe[column]
        missing_percentage = _percentage(
            int(series.isna().sum()),
            len(dataframe),
        )

        if series.notna().sum() == 0:
            problematic_columns.add(column)
            issues.append(
                _issue(
                    "EMPTY_FEATURE",
                    "high",
                    name,
                    "The feature contains no observed values.",
                    column=str(column),
                )
            )

        elif series.nunique(dropna=True) <= 1:
            problematic_columns.add(column)
            issues.append(
                _issue(
                    "CONSTANT_FEATURE",
                    "medium",
                    name,
                    "The feature has no observed variation.",
                    column=str(column),
                )
            )

        elif missing_percentage > MAX_MISSING_PERCENTAGE:
            problematic_columns.add(column)
            issues.append(
                _issue(
                    "HIGH_MISSING_FEATURE",
                    "medium",
                    name,
                    "More than half of this feature is missing.",
                    column=str(column),
                    evidence={
                        "missing_percentage": missing_percentage,
                    },
                )
            )

    problematic_percentage = _percentage(
        len(problematic_columns),
        len(features),
    )

    return _result(
        name,
        _score_from_penalty(problematic_percentage),
        (
            "Preliminary feature suitability measures empty, "
            "constant and high-missing features. Algorithm-specific "
            "feature checks are not yet included."
        ),
        issues=issues,
        metrics={
            "feature_count": len(features),
            "problematic_feature_count": len(problematic_columns),
            "problematic_percentage": problematic_percentage,
        },
    )


def score_target_suitability(
    dataframe: pd.DataFrame,
    *,
    task: MLTask,
    target_column: str | None,
) -> ReadinessDimensionResult | None:
    """Run preliminary target checks for supervised ML tasks.

    Clustering has no target; return None so the assessment
    engine can apply a task-specific configuration later.
    """

    name = "target_suitability"

    if task == "clustering":
        return None

    if not target_column or target_column not in dataframe.columns:
        return _result(
            name,
            0.0,
            "The required target column is missing.",
            issues=[
                _issue(
                    "MISSING_TARGET",
                    "critical",
                    name,
                    "The selected target column does not exist.",
                )
            ],
        )

    target = dataframe[target_column]
    missing_percentage = _percentage(
        int(target.isna().sum()),
        len(target),
    )

    issues = []

    if missing_percentage:
        issues.append(
            _issue(
                "MISSING_TARGET_VALUES",
                "high",
                name,
                "The target contains missing values.",
                column=target_column,
                evidence={
                    "missing_percentage": missing_percentage,
                },
            )
        )

    unique_count = int(target.nunique(dropna=True))

    if unique_count <= 1:
        issues.append(
            _issue(
                "CONSTANT_TARGET",
                "critical",
                name,
                "The target has fewer than two distinct observed values.",
                column=target_column,
            )
        )

    # Regression and forecasting generally require numeric
    # targets in the initial supported workflow.
    incompatible_dtype = (
        task in ("regression", "forecasting")
        and not pd.api.types.is_numeric_dtype(target)
    )

    if incompatible_dtype:
        issues.append(
            _issue(
                "NON_NUMERIC_TARGET",
                "high",
                name,
                "The selected task requires a numeric target.",
                column=target_column,
            )
        )

    score = _score_from_penalty(missing_percentage)

    if unique_count <= 1 or incompatible_dtype:
        score = 0.0

    return _result(
        name,
        score,
        (
            "Preliminary target suitability checks target "
            "availability, missing values, variation and numeric "
            "compatibility where required."
        ),
        issues=issues,
        metrics={
            "target_column": target_column,
            "target_missing_percentage": missing_percentage,
            "target_unique_count": unique_count,
            "target_dtype": str(target.dtype),
        },
    )


def score_data_sufficiency(
    dataframe: pd.DataFrame,
    *,
    task: MLTask,
    target_column: str | None = None,
) -> ReadinessDimensionResult:
    """Apply transparent preliminary sample-count thresholds.

    These thresholds are heuristics, not guarantees that a
    particular model will perform well.
    """

    name = "data_sufficiency"
    row_count = len(dataframe)
    issues = []

    # The initial threshold is deliberately explicit.
    score = min(
        100.0,
        round(100.0 * row_count / MIN_ROWS, 2),
    )

    if row_count < MIN_ROWS:
        issues.append(
            _issue(
                "LOW_SAMPLE_COUNT",
                "medium",
                name,
                "The dataset has fewer than 100 rows.",
                recommendation=(
                    "Assess sample requirements for the intended "
                    "algorithm and validation strategy."
                ),
                evidence={
                    "row_count": row_count,
                    "initial_threshold": MIN_ROWS,
                },
            )
        )

    # Classification needs sufficient examples per class.
    minimum_class_count = None

    if (
        task == "classification"
        and target_column
        and target_column in dataframe.columns
    ):
        counts = dataframe[target_column].value_counts(
            dropna=True
        )

        if not counts.empty:
            minimum_class_count = int(counts.min())

            if minimum_class_count < MIN_CLASS_COUNT:
                issues.append(
                    _issue(
                        "SMALL_CLASS",
                        "high",
                        name,
                        "At least one target class has fewer than 10 examples.",
                        recommendation=(
                            "Review class balance and the planned "
                            "train/validation split."
                        ),
                        evidence={
                            "minimum_class_count": minimum_class_count,
                            "initial_threshold": MIN_CLASS_COUNT,
                        },
                    )
                )

                score = min(
                    score,
                    round(
                        100.0
                        * minimum_class_count
                        / MIN_CLASS_COUNT,
                        2,
                    ),
                )

    return _result(
        name,
        score,
        (
            "Preliminary sample sufficiency uses explicit row-count "
            "and, for classification, minimum-class-count heuristics. "
            "Task-specific requirements will refine this score."
        ),
        issues=issues,
        metrics={
            "row_count": row_count,
            "initial_minimum_rows": MIN_ROWS,
            "minimum_class_count": minimum_class_count,
        },
    )


def evaluate_dimensions(
    dataframe: pd.DataFrame,
    *,
    task: MLTask,
    target_column: str | None = None,
    excluded_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate available preliminary dimensions.

    The result is not a complete ML readiness assessment.
    Pending dimensions are explicitly listed so the caller
    cannot mistake partial coverage for a final score.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("A pandas DataFrame is required.")

    if dataframe.empty:
        raise ValueError("Cannot assess an empty dataset.")

    if not dataframe.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")

    if task not in (
        "classification",
        "regression",
        "forecasting",
        "clustering",
    ):
        raise ValueError(f"Unsupported ML task: {task}")

    if task != "clustering":
        if not target_column:
            raise ValueError(
                "A target column is required for supervised tasks."
            )

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column not found: {target_column}"
            )

    results = [
        score_completeness(dataframe),
        score_validity_consistency(dataframe),
        score_record_integrity(dataframe),
        score_feature_suitability(
            dataframe,
            target_column=target_column,
            excluded_columns=excluded_columns,
        ),
    ]

    target_result = score_target_suitability(
        dataframe,
        task=task,
        target_column=target_column,
    )

    pending = []

    if target_result is not None:
        results.append(target_result)
    else:
        pending.append({
            "dimension": "target_suitability",
            "reason": (
                "Clustering requires a task-specific replacement "
                "for supervised target suitability."
            ),
        })

    results.append(
        score_data_sufficiency(
            dataframe,
            task=task,
            target_column=target_column,
        )
    )

    # Do not assign a score until leakage and validation
    # checks are actually implemented.
    pending.append({
        "dimension": "leakage_validation",
        "reason": (
            "Requires the dedicated leakage and validation engine."
        ),
    })

    return {
        "task": task,
        "status": "partial",
        "dimensions": [
            result.model_dump(mode="json")
            for result in results
        ],
        "pending_dimensions": pending,
        "overall_score": None,
    }
