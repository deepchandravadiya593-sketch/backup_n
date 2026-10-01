
"""NUMPA v2 - Classification-specific ML readiness checks.

Provides deterministic diagnostics for binary and multiclass
classification without fitting or evaluating an ML model.

All thresholds are preliminary, explicit heuristics.
They do not guarantee model quality or generalization.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.readiness import (
    ReadinessIssue,
    ValidationStrategy,
)


# Default classification assessment thresholds.
MIN_CLASS_COUNT = 10
MIN_TRAIN_CLASS_COUNT = 2
MIN_VALIDATION_CLASS_COUNT = 1
MIN_TOTAL_ROWS = 100

# Warn when the largest class has more than ten times
# as many observations as the smallest class.
IMBALANCE_RATIO_THRESHOLD = 10.0

# A very high number of distinct numeric target values
# can indicate that a regression target was selected.
HIGH_CARDINALITY_MIN_CLASSES = 20
HIGH_CARDINALITY_RATIO = 0.50

DEFAULT_VALIDATION_FRACTION = 0.20


def _percentage(count: int, total: int) -> float:
    """Return a percentage without division by zero."""

    if total <= 0:
        return 0.0

    return round(100.0 * count / total, 4)


def _issue(
    code: str,
    severity: str,
    message: str,
    *,
    recommendation: str | None = None,
    evidence: dict[str, Any] | None = None,
    column: str | None = None,
) -> ReadinessIssue:
    """Create an issue compatible with the shared API schema."""

    return ReadinessIssue(
        code=code,
        severity=severity,
        dimension="target_suitability",
        message=message,
        recommendation=recommendation,
        evidence=evidence or {},
        column=column,
    )


def _safe_class_label(value: Any) -> str:
    """Represent a class label safely for JSON reports."""

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return str(value)


def _validate_fraction(
    validation_fraction: float,
) -> None:
    """Require a validation fraction strictly between zero and one."""

    if (
        isinstance(validation_fraction, bool)
        or not isinstance(
            validation_fraction,
            (int, float),
        )
        or not math.isfinite(float(validation_fraction))
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError(
            "validation_fraction must be a finite number "
            "strictly between 0 and 1."
        )


def _class_distribution(
    target: pd.Series,
) -> list[dict[str, Any]]:
    """Build a deterministic class-frequency report."""

    observed = target.dropna()

    if observed.empty:
        return []

    try:
        counts = observed.value_counts(dropna=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The target contains values that cannot be "
            "counted as classification labels."
        ) from exc

    distribution = []

    for label, count in counts.items():
        distribution.append({
            "label": _safe_class_label(label),
            "count": int(count),
            "percentage": _percentage(
                int(count),
                len(observed),
            ),
        })

    # Sort by frequency, then label, so reports are reproducible.
    distribution.sort(
        key=lambda item: (
            -item["count"],
            item["label"],
        )
    )

    return distribution


def _check_stratified_split(
    *,
    class_count: int,
    smallest_class_count: int,
    observed_rows: int,
    validation_fraction: float,
) -> tuple[bool, list[ReadinessIssue]]:
    """Check basic feasibility of a stratified holdout split.

    This is a conservative preflight check, not a substitute
    for executing the actual splitter.
    """

    issues: list[ReadinessIssue] = []

    if class_count < 2:
        return False, issues

    validation_rows = math.ceil(
        observed_rows * validation_fraction
    )

    training_rows = observed_rows - validation_rows

    # Each split needs enough total rows to represent
    # every observed class at least once.
    if (
        validation_rows < class_count
        or training_rows < class_count
    ):
        issues.append(
            _issue(
                "STRATIFIED_SPLIT_TOO_SMALL",
                "high",
                (
                    "The requested split does not have enough "
                    "rows to represent every class in both sets."
                ),
                recommendation=(
                    "Adjust the validation fraction or obtain "
                    "more observations for the smaller classes."
                ),
                evidence={
                    "class_count": class_count,
                    "training_rows": training_rows,
                    "validation_rows": validation_rows,
                },
            )
        )

    # Standard stratified splitting needs at least two
    # observations in every class.
    if smallest_class_count < 2:
        issues.append(
            _issue(
                "STRATIFIED_SPLIT_SINGLETON_CLASS",
                "high",
                (
                    "At least one class contains fewer than "
                    "two observations."
                ),
                recommendation=(
                    "Obtain additional examples or reconsider "
                    "how the rare class is defined."
                ),
                evidence={
                    "smallest_class_count": smallest_class_count,
                },
            )
        )

    return len(issues) == 0, issues


def assess_classification(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    validation_strategy: ValidationStrategy = "auto",
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    intended_algorithm: str | None = None,
) -> dict[str, Any]:
    """Assess preliminary classification readiness.

    Returns task-specific scores, class statistics,
    validation feasibility and actionable issues.

    Scores are task-specific diagnostic measurements;
    the main engine must decide how to combine them with
    other dimensions and outstanding checks.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "A pandas DataFrame is required."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot assess an empty dataset."
        )

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Dataset column names must be unique."
        )

    if not target_column or target_column not in dataframe.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    _validate_fraction(validation_fraction)

    supported_strategies = {
        "auto",
        "random_split",
        "stratified_split",
        "time_series_split",
        "group_split",
        "cross_validation",
        "none",
    }

    if validation_strategy not in supported_strategies:
        raise ValueError(
            "Unsupported validation strategy: "
            f"{validation_strategy}"
        )

    target = dataframe[target_column]

    total_rows = int(len(dataframe))
    missing_count = int(target.isna().sum())
    observed_rows = total_rows - missing_count

    issues: list[ReadinessIssue] = []

    if missing_count:
        issues.append(
            _issue(
                "CLASSIFICATION_MISSING_TARGET",
                "high",
                "The classification target contains missing values.",
                column=target_column,
                recommendation=(
                    "Review rows with missing target labels "
                    "before creating the train/validation split."
                ),
                evidence={
                    "missing_count": missing_count,
                    "missing_percentage": _percentage(
                        missing_count,
                        total_rows,
                    ),
                },
            )
        )

    distribution = _class_distribution(target)
    class_count = len(distribution)

    smallest_class_count = (
        min(item["count"] for item in distribution)
        if distribution
        else 0
    )

    largest_class_count = (
        max(item["count"] for item in distribution)
        if distribution
        else 0
    )

    imbalance_ratio = (
        round(
            largest_class_count / smallest_class_count,
            4,
        )
        if smallest_class_count > 0
        else None
    )

    classification_type = (
        "binary"
        if class_count == 2
        else "multiclass"
        if class_count > 2
        else "invalid"
    )

    # Start with a complete target suitability score,
    # then apply transparent penalties.
    target_score = 100.0

    if total_rows > 0:
        target_score -= (
            100.0 * missing_count / total_rows
        )

    if class_count < 2:
        target_score = 0.0

        issues.append(
            _issue(
                "INSUFFICIENT_TARGET_CLASSES",
                "critical",
                (
                    "Classification requires at least two "
                    "observed target classes."
                ),
                column=target_column,
                recommendation=(
                    "Select a target containing at least "
                    "two distinct classes."
                ),
                evidence={
                    "class_count": class_count,
                },
            )
        )

    if (
        class_count >= HIGH_CARDINALITY_MIN_CLASSES
        and observed_rows > 0
        and class_count / observed_rows
        >= HIGH_CARDINALITY_RATIO
    ):
        issues.append(
            _issue(
                "HIGH_CARDINALITY_TARGET",
                "medium",
                (
                    "The target has many distinct labels "
                    "relative to the number of observations."
                ),
                column=target_column,
                recommendation=(
                    "Confirm that the target represents "
                    "classification labels rather than a "
                    "continuous measurement or identifier."
                ),
                evidence={
                    "class_count": class_count,
                    "observed_rows": observed_rows,
                    "cardinality_ratio": round(
                        class_count / observed_rows,
                        4,
                    ),
                },
            )
        )

    if (
        class_count >= 2
        and imbalance_ratio is not None
        and imbalance_ratio >= IMBALANCE_RATIO_THRESHOLD
    ):
        issues.append(
            _issue(
                "SEVERE_CLASS_IMBALANCE",
                "high",
                (
                    "The largest class has at least ten "
                    "times as many examples as the smallest."
                ),
                column=target_column,
                recommendation=(
                    "Use suitable evaluation metrics and "
                    "consider class weighting or other "
                    "training-only balancing methods."
                ),
                evidence={
                    "imbalance_ratio": imbalance_ratio,
                    "threshold": IMBALANCE_RATIO_THRESHOLD,
                },
            )
        )

    # Preliminary sufficiency uses both total rows
    # and the number of examples in the rarest class.
    row_sufficiency = min(
        100.0,
        100.0 * total_rows / MIN_TOTAL_ROWS,
    )

    class_sufficiency = min(
        100.0,
        100.0
        * smallest_class_count
        / MIN_CLASS_COUNT,
    )

    data_sufficiency_score = round(
        min(row_sufficiency, class_sufficiency),
        2,
    )

    if (
        class_count >= 2
        and smallest_class_count < MIN_CLASS_COUNT
    ):
        issues.append(
            _issue(
                "INSUFFICIENT_MINORITY_CLASS_SAMPLES",
                "high",
                (
                    "The smallest class has fewer than "
                    "the preliminary minimum of ten examples."
                ),
                column=target_column,
                recommendation=(
                    "Review minority-class sample requirements "
                    "for the chosen algorithm and evaluation plan."
                ),
                evidence={
                    "smallest_class_count": smallest_class_count,
                    "minimum": MIN_CLASS_COUNT,
                },
            )
        )

    # Auto mode recommends stratification, but the
    # actual strategy must be selected by the caller.
    suggested_strategy = (
        "stratified_split"
        if validation_strategy == "auto"
        else validation_strategy
    )

    stratified_feasible, stratified_issues = (
        _check_stratified_split(
            class_count=class_count,
            smallest_class_count=smallest_class_count,
            observed_rows=observed_rows,
            validation_fraction=validation_fraction,
        )
    )

    # Only report stratification-specific failures when
    # stratification was selected or suggested.
    if suggested_strategy == "stratified_split":
        issues.extend(stratified_issues)

    if validation_strategy == "random_split":
        issues.append(
            _issue(
                "UNSTRATIFIED_CLASSIFICATION_SPLIT",
                "low",
                (
                    "A random split may produce different "
                    "class proportions across the two sets."
                ),
                recommendation=(
                    "Consider stratification when preserving "
                    "class proportions is appropriate."
                ),
            )
        )

    # Group and time-series validation need their own
    # column-role and split-leakage checks.
    validation_requires_external_checks = (
        suggested_strategy
        in {
            "group_split",
            "time_series_split",
            "cross_validation",
        }
    )

    # Count critical issues separately from warnings.
    critical_count = sum(
        issue.severity == "critical"
        for issue in issues
    )

    high_count = sum(
        issue.severity == "high"
        for issue in issues
    )

    return {
        "task": "classification",
        "target_column": target_column,
        "intended_algorithm": intended_algorithm,
        "classification_type": classification_type,
        "total_rows": total_rows,
        "observed_target_rows": observed_rows,
        "missing_target_count": missing_count,
        "class_count": class_count,
        "class_distribution": distribution,
        "smallest_class_count": smallest_class_count,
        "largest_class_count": largest_class_count,
        "imbalance_ratio": imbalance_ratio,
        "scores": {
            "target_suitability": round(
                max(0.0, min(100.0, target_score)),
                2,
            ),
            "data_sufficiency": data_sufficiency_score,
        },
        "validation": {
            "requested_strategy": validation_strategy,
            "suggested_strategy": suggested_strategy,
            "validation_fraction": validation_fraction,
            "stratified_split_feasible": (
                stratified_feasible
            ),
            "requires_external_checks": (
                validation_requires_external_checks
            ),
            "actual_split_checked": False,
        },
        "issue_count": len(issues),
        "critical_issue_count": critical_count,
        "high_issue_count": high_count,
        "issues": [
            issue.model_dump(mode="json")
            for issue in issues
        ],
        "limitations": [
            (
                "Sample-count thresholds are preliminary "
                "heuristics, not model-performance guarantees."
            ),
            (
                "The actual train/validation split has "
                "not been executed or inspected."
            ),
            (
                "Algorithm-specific requirements and "
                "preprocessing leakage are not evaluated here."
            ),
        ],
    }
