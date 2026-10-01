
"""NUMPA v2 - Regression-specific ML readiness checks.

Provides deterministic diagnostics for supervised regression.

Checks:
- Numeric target compatibility and missing values
- Target variation and non-finite values
- Potential target outliers
- Preliminary sample sufficiency
- Basic validation strategy feasibility

Thresholds are documented heuristics, not guarantees
of model performance.
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


# Preliminary regression assessment thresholds.
MIN_TOTAL_ROWS = 100
MIN_TRAIN_ROWS = 20
MIN_VALIDATION_ROWS = 5
DEFAULT_VALIDATION_FRACTION = 0.20

# IQR-based target outlier detection.
OUTLIER_IQR_MULTIPLIER = 1.5
HIGH_OUTLIER_PERCENTAGE = 10.0


def _percentage(
    count: int,
    total: int,
) -> float:
    """Calculate a percentage safely."""

    if total <= 0:
        return 0.0

    return round(
        100.0 * count / total,
        4,
    )


def _issue(
    code: str,
    severity: str,
    message: str,
    *,
    column: str | None = None,
    recommendation: str | None = None,
    evidence: dict[str, Any] | None = None,
    dimension: str = "target_suitability",
) -> ReadinessIssue:
    """Create an explainable regression issue."""

    return ReadinessIssue(
        code=code,
        severity=severity,
        dimension=dimension,
        message=message,
        column=column,
        recommendation=recommendation,
        evidence=evidence or {},
    )


def _validate_fraction(
    fraction: float,
) -> None:
    """Require a finite validation fraction between zero and one."""

    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 < fraction < 1.0
    ):
        raise ValueError(
            "validation_fraction must be a finite number "
            "strictly between 0 and 1."
        )


def _target_outliers(
    values: pd.Series,
) -> dict[str, Any]:
    """Detect potential target outliers using the IQR method.

    This is diagnostic only. Target outliers may be genuine
    observations and must not be removed automatically.
    """

    clean = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(clean) < 4:
        return {
            "checked": False,
            "reason": "At least four finite values are required.",
            "count": 0,
            "percentage": None,
            "lower_bound": None,
            "upper_bound": None,
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        return {
            "checked": False,
            "reason": (
                "The interquartile range is zero; "
                "IQR outlier detection is inconclusive."
            ),
            "count": 0,
            "percentage": None,
            "lower_bound": None,
            "upper_bound": None,
        }

    lower_bound = (
        q1 - OUTLIER_IQR_MULTIPLIER * iqr
    )
    upper_bound = (
        q3 + OUTLIER_IQR_MULTIPLIER * iqr
    )

    outlier_mask = (
        (clean < lower_bound)
        | (clean > upper_bound)
    )

    count = int(outlier_mask.sum())

    return {
        "checked": True,
        "reason": None,
        "count": count,
        "percentage": _percentage(
            count,
            len(clean),
        ),
        "lower_bound": round(lower_bound, 6),
        "upper_bound": round(upper_bound, 6),
    }


def _validation_feasibility(
    *,
    observed_rows: int,
    strategy: ValidationStrategy,
    validation_fraction: float,
) -> dict[str, Any]:
    """Evaluate basic holdout feasibility.

    Group, temporal and cross-validation checks require
    additional configuration and actual split inspection.
    """

    suggested_strategy = (
        "random_split"
        if strategy == "auto"
        else strategy
    )

    validation_rows = math.ceil(
        observed_rows * validation_fraction
    )
    training_rows = (
        observed_rows - validation_rows
    )

    holdout_feasible = (
        training_rows >= MIN_TRAIN_ROWS
        and validation_rows >= MIN_VALIDATION_ROWS
    )

    requires_external_checks = (
        suggested_strategy
        in {
            "group_split",
            "time_series_split",
            "cross_validation",
        }
    )

    return {
        "requested_strategy": strategy,
        "suggested_strategy": suggested_strategy,
        "validation_fraction": validation_fraction,
        "estimated_training_rows": training_rows,
        "estimated_validation_rows": validation_rows,
        "holdout_feasible": holdout_feasible,
        "requires_external_checks": requires_external_checks,
        "actual_split_checked": False,
    }


def assess_regression(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    validation_strategy: ValidationStrategy = "auto",
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    intended_algorithm: str | None = None,
) -> dict[str, Any]:
    """Generate a preliminary regression-readiness report.

    Returns target diagnostics, preliminary dimension scores,
    validation feasibility and explainable issues.

    The result must not be presented as a complete ML
    readiness assessment.
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

    if (
        not target_column
        or target_column not in dataframe.columns
    ):
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

    issues: list[ReadinessIssue] = []

    # Regression requires a numeric target in NUMPA's
    # initial supported workflow. Do not silently coerce
    # text columns into numeric targets.
    numeric_target = pd.api.types.is_numeric_dtype(
        target
    )

    if not numeric_target:
        issues.append(
            _issue(
                "NON_NUMERIC_REGRESSION_TARGET",
                "critical",
                "The regression target is not numeric.",
                column=target_column,
                recommendation=(
                    "Choose a numeric target or explicitly "
                    "convert the column after reviewing its values."
                ),
                evidence={
                    "target_dtype": str(target.dtype),
                },
            )
        )

        return {
            "task": "regression",
            "target_column": target_column,
            "intended_algorithm": intended_algorithm,
            "total_rows": total_rows,
            "observed_target_rows": 0,
            "target_statistics": None,
            "outliers": {
                "checked": False,
                "reason": "The target is not numeric.",
            },
            "scores": {
                "target_suitability": 0.0,
                "data_sufficiency": 0.0,
            },
            "validation": {
                "requested_strategy": validation_strategy,
                "suggested_strategy": None,
                "actual_split_checked": False,
                "requires_external_checks": True,
            },
            "issue_count": len(issues),
            "critical_issue_count": 1,
            "high_issue_count": 0,
            "issues": [
                issue.model_dump(mode="json")
                for issue in issues
            ],
            "limitations": [
                "A numeric target is required before further scoring.",
            ],
        }

    # Convert numeric values to floating point for finite
    # checks without modifying the source DataFrame.
    numeric = pd.to_numeric(
        target,
        errors="coerce",
    ).astype(float)

    missing_count = int(
        numeric.isna().sum()
    )

    infinite_count = int(
        np.isinf(numeric.to_numpy()).sum()
    )

    finite = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    observed_rows = int(len(finite))
    unique_count = int(
        finite.nunique()
    )

    if missing_count:
        issues.append(
            _issue(
                "MISSING_REGRESSION_TARGET",
                "high",
                "The regression target contains missing values.",
                column=target_column,
                recommendation=(
                    "Review missing target observations before "
                    "creating the training and validation sets."
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

    if infinite_count:
        issues.append(
            _issue(
                "INFINITE_REGRESSION_TARGET",
                "high",
                "The regression target contains infinite values.",
                column=target_column,
                recommendation=(
                    "Investigate infinite values before training."
                ),
                evidence={
                    "infinite_count": infinite_count,
                },
            )
        )

    if unique_count <= 1:
        issues.append(
            _issue(
                "CONSTANT_REGRESSION_TARGET",
                "critical",
                (
                    "The target has fewer than two distinct "
                    "finite values."
                ),
                column=target_column,
                recommendation=(
                    "Choose a target with meaningful variation."
                ),
                evidence={
                    "unique_count": unique_count,
                },
            )
        )

    # A target with few distinct values can be valid for
    # regression, so report it as informational rather
    # than automatically rejecting it.
    if 2 <= unique_count <= 5:
        issues.append(
            _issue(
                "LOW_TARGET_CARDINALITY",
                "info",
                (
                    "The numeric target has relatively few "
                    "distinct observed values."
                ),
                column=target_column,
                recommendation=(
                    "Confirm that regression matches the "
                    "intended prediction objective."
                ),
                evidence={
                    "unique_count": unique_count,
                },
            )
        )

    outliers = _target_outliers(
        numeric
    )

    if (
        outliers["checked"]
        and outliers["percentage"]
        >= HIGH_OUTLIER_PERCENTAGE
    ):
        issues.append(
            _issue(
                "HIGH_TARGET_OUTLIER_RATE",
                "medium",
                (
                    "A substantial proportion of finite "
                    "target values fall outside the IQR bounds."
                ),
                column=target_column,
                recommendation=(
                    "Investigate unusual target observations. "
                    "Do not remove them without domain justification."
                ),
                evidence={
                    "outlier_count": outliers["count"],
                    "outlier_percentage": (
                        outliers["percentage"]
                    ),
                },
            )
        )

    target_statistics = None

    if observed_rows:
        target_statistics = {
            "minimum": float(finite.min()),
            "maximum": float(finite.max()),
            "mean": float(finite.mean()),
            "median": float(finite.median()),
            "standard_deviation": (
                float(finite.std(ddof=0))
            ),
            "unique_count": unique_count,
        }

    # Target suitability measures the proportion of
    # finite, observed target values.
    target_score = (
        100.0 * observed_rows / total_rows
    )

    # A constant or entirely missing target cannot
    # support ordinary supervised regression.
    if unique_count <= 1:
        target_score = 0.0

    target_score = round(
        max(0.0, min(100.0, target_score)),
        2,
    )

    # Sample sufficiency is a preliminary heuristic.
    data_sufficiency_score = round(
        min(
            100.0,
            100.0
            * observed_rows
            / MIN_TOTAL_ROWS,
        ),
        2,
    )

    if observed_rows < MIN_TOTAL_ROWS:
        issues.append(
            _issue(
                "LOW_REGRESSION_SAMPLE_COUNT",
                "medium",
                (
                    "The dataset has fewer than 100 "
                    "finite target observations."
                ),
                dimension="data_sufficiency",
                recommendation=(
                    "Review sample requirements for the "
                    "chosen algorithm and validation strategy."
                ),
                evidence={
                    "observed_rows": observed_rows,
                    "initial_threshold": MIN_TOTAL_ROWS,
                },
            )
        )

    validation = _validation_feasibility(
        observed_rows=observed_rows,
        strategy=validation_strategy,
        validation_fraction=validation_fraction,
    )

    if (
        validation["suggested_strategy"]
        in {"random_split", "stratified_split"}
        and not validation["holdout_feasible"]
    ):
        issues.append(
            _issue(
                "REGRESSION_HOLDOUT_TOO_SMALL",
                "high",
                (
                    "The requested holdout would leave too "
                    "few training or validation observations."
                ),
                dimension="data_sufficiency",
                recommendation=(
                    "Obtain more observations or review "
                    "the validation strategy."
                ),
                evidence={
                    "estimated_training_rows": (
                        validation["estimated_training_rows"]
                    ),
                    "estimated_validation_rows": (
                        validation["estimated_validation_rows"]
                    ),
                },
            )
        )

    if validation_strategy == "stratified_split":
        issues.append(
            _issue(
                "REGRESSION_STRATIFICATION_REVIEW",
                "medium",
                (
                    "Ordinary stratified splitting expects "
                    "discrete class labels, not continuous targets."
                ),
                dimension="leakage_validation",
                recommendation=(
                    "Consider random, group-aware or chronological "
                    "validation as appropriate. If target binning "
                    "is intended, define and review it explicitly."
                ),
            )
        )

    critical_count = sum(
        issue.severity == "critical"
        for issue in issues
    )

    high_count = sum(
        issue.severity == "high"
        for issue in issues
    )

    return {
        "task": "regression",
        "target_column": target_column,
        "intended_algorithm": intended_algorithm,
        "total_rows": total_rows,
        "observed_target_rows": observed_rows,
        "missing_target_count": missing_count,
        "infinite_target_count": infinite_count,
        "target_statistics": target_statistics,
        "outliers": outliers,
        "scores": {
            "target_suitability": target_score,
            "data_sufficiency": data_sufficiency_score,
        },
        "validation": validation,
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
                "IQR outlier detection is diagnostic and "
                "does not establish that values are erroneous."
            ),
            (
                "The actual validation split and "
                "preprocessing pipeline have not been audited."
            ),
            (
                "Algorithm-specific regression requirements "
                "are not yet evaluated."
            ),
        ],
    }
