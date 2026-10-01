
"""NUMPA v2 - Time-series forecasting readiness checks.

Provides deterministic diagnostics for chronological datasets.

The module checks timestamps, frequency, gaps, target values,
history length and validation suitability without fitting a model.

Scores are preliminary heuristics, not performance guarantees.
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


# Explicit preliminary thresholds.
MIN_HISTORY_ROWS = 100
MIN_TRAIN_ROWS = 30
MIN_VALIDATION_ROWS = 10
DEFAULT_VALIDATION_FRACTION = 0.20

# At least three distinct timestamps are required to
# calculate and compare observation intervals.
MIN_FREQUENCY_TIMESTAMPS = 3


def _percentage(count: int, total: int) -> float:
    """Return a percentage without dividing by zero."""

    if total <= 0:
        return 0.0

    return round(100.0 * count / total, 4)


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
    """Create an issue compatible with NUMPA's shared schema."""

    return ReadinessIssue(
        code=code,
        severity=severity,
        dimension=dimension,
        message=message,
        column=column,
        recommendation=recommendation,
        evidence=evidence or {},
    )


def _validate_fraction(value: float) -> None:
    """Require a finite validation fraction between zero and one."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < value < 1.0
    ):
        raise ValueError(
            "validation_fraction must be a finite number "
            "strictly between 0 and 1."
        )


def _analyze_frequency(
    timestamps: pd.Series,
) -> dict[str, Any]:
    """Examine intervals between distinct chronological timestamps.

    The most common interval is treated as a candidate frequency.
    Missing periods are estimated only when all observed intervals
    are exact multiples of that candidate.
    """

    unique_dates = (
        timestamps.dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_dates) < MIN_FREQUENCY_TIMESTAMPS:
        return {
            "checked": False,
            "reason": "At least three distinct timestamps are required.",
            "regular": None,
            "candidate_interval_seconds": None,
            "missing_periods": None,
            "distinct_interval_count": None,
        }

    # Integer nanosecond differences avoid ambiguous string
    # comparisons and preserve sub-second intervals.
    nanoseconds = unique_dates.astype("int64").to_numpy()
    intervals = np.diff(nanoseconds)

    positive_intervals = intervals[intervals > 0]

    if positive_intervals.size == 0:
        return {
            "checked": False,
            "reason": "No positive timestamp intervals are available.",
            "regular": None,
            "candidate_interval_seconds": None,
            "missing_periods": None,
            "distinct_interval_count": None,
        }

    values, counts = np.unique(
        positive_intervals,
        return_counts=True,
    )

    # Prefer the shortest interval when frequencies tie.
    most_common_count = counts.max()
    candidate = int(
        values[counts == most_common_count].min()
    )

    regular = bool(
        np.all(positive_intervals == candidate)
    )

    # Estimate missing periods only when the candidate
    # divides every observed interval exactly.
    exact_multiples = bool(
        np.all(positive_intervals % candidate == 0)
    )

    missing_periods = (
        int(
            sum(
                int(interval // candidate) - 1
                for interval in positive_intervals
            )
        )
        if exact_multiples
        else None
    )

    return {
        "checked": True,
        "reason": None,
        "regular": regular,
        "candidate_interval_seconds": round(
            candidate / 1_000_000_000,
            6,
        ),
        "missing_periods": missing_periods,
        "distinct_interval_count": int(len(values)),
    }


def _analyze_series(
    dataframe: pd.DataFrame,
    timestamps: pd.Series,
    *,
    target_column: str,
    datetime_column: str,
    series_label: str,
) -> tuple[dict[str, Any], list[ReadinessIssue]]:
    """Analyze one time series independently."""

    issues: list[ReadinessIssue] = []

    valid_dates = timestamps.dropna()
    duplicate_count = int(
        valid_dates.duplicated(keep="first").sum()
    )

    # Check the order in which records were uploaded.
    chronological = bool(
        valid_dates.is_monotonic_increasing
    )

    if duplicate_count:
        issues.append(
            _issue(
                "DUPLICATE_TIMESTAMPS",
                "high",
                "record_integrity",
                "Multiple observations share the same timestamp.",
                column=datetime_column,
                recommendation=(
                    "Determine whether observations should be "
                    "aggregated or represent separate series."
                ),
                evidence={
                    "series": series_label,
                    "duplicate_count": duplicate_count,
                },
            )
        )

    if not chronological:
        issues.append(
            _issue(
                "UNSORTED_TIMESTAMPS",
                "medium",
                "validity_consistency",
                "Observations are not in chronological order.",
                column=datetime_column,
                recommendation=(
                    "Sort each series chronologically before "
                    "creating time-dependent features or splits."
                ),
                evidence={"series": series_label},
            )
        )

    frequency = _analyze_frequency(timestamps)

    if frequency["checked"] and not frequency["regular"]:
        issues.append(
            _issue(
                "IRREGULAR_TIME_INTERVALS",
                "medium",
                "validity_consistency",
                "Observation intervals are not uniform.",
                column=datetime_column,
                recommendation=(
                    "Review the expected sampling frequency. "
                    "Irregular intervals may be intentional."
                ),
                evidence={
                    "series": series_label,
                    "candidate_interval_seconds": (
                        frequency["candidate_interval_seconds"]
                    ),
                },
            )
        )

    if (
        frequency["missing_periods"] is not None
        and frequency["missing_periods"] > 0
    ):
        issues.append(
            _issue(
                "MISSING_TIME_PERIODS",
                "medium",
                "completeness",
                "Expected observation periods are missing.",
                column=datetime_column,
                recommendation=(
                    "Investigate missing periods before deciding "
                    "whether to interpolate, impute or retain gaps."
                ),
                evidence={
                    "series": series_label,
                    "missing_periods": frequency["missing_periods"],
                },
            )
        )

    target = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    )

    finite_target = target.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    result = {
        "series": series_label,
        "rows": int(len(dataframe)),
        "valid_timestamps": int(valid_dates.size),
        "duplicate_timestamps": duplicate_count,
        "chronologically_sorted": chronological,
        "first_timestamp": (
            valid_dates.min().isoformat()
            if not valid_dates.empty
            else None
        ),
        "last_timestamp": (
            valid_dates.max().isoformat()
            if not valid_dates.empty
            else None
        ),
        "frequency": frequency,
        "finite_target_rows": int(len(finite_target)),
    }

    return result, issues


def assess_forecasting(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    datetime_column: str,
    group_column: str | None = None,
    validation_strategy: ValidationStrategy = "auto",
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    intended_algorithm: str | None = None,
) -> dict[str, Any]:
    """Generate a preliminary forecasting-readiness report.

    Each group is evaluated as an independent series when
    group_column is provided.

    The report identifies temporal risks but does not inspect
    actual train/validation row assignments.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("A pandas DataFrame is required.")

    if dataframe.empty:
        raise ValueError("Cannot assess an empty dataset.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Dataset column names must be unique."
        )

    if not target_column or target_column not in dataframe.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    if not datetime_column or datetime_column not in dataframe.columns:
        raise ValueError(
            f"Datetime column not found: {datetime_column}"
        )

    if group_column and group_column not in dataframe.columns:
        raise ValueError(
            f"Group column not found: {group_column}"
        )

    if len({
        column
        for column in (
            target_column,
            datetime_column,
            group_column,
        )
        if column is not None
    }) != (
        3 if group_column else 2
    ):
        raise ValueError(
            "Target, datetime and group columns must be distinct."
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
            f"Unsupported validation strategy: {validation_strategy}"
        )

    issues: list[ReadinessIssue] = []
    total_rows = int(len(dataframe))

    # Do not silently treat text or identifiers as
    # valid numeric forecasting targets.
    numeric_target = pd.api.types.is_numeric_dtype(
        dataframe[target_column]
    )

    if not numeric_target:
        issues.append(
            _issue(
                "NON_NUMERIC_FORECAST_TARGET",
                "critical",
                "target_suitability",
                "The forecasting target is not numeric.",
                column=target_column,
                recommendation=(
                    "Select a numeric target or explicitly "
                    "convert the column after reviewing its values."
                ),
            )
        )

    # Parsing is diagnostic only and does not modify the dataset.
    # utc=True makes mixed timezone offsets comparable.
    parsed_dates = pd.to_datetime(
        dataframe[datetime_column],
        errors="coerce",
        utc=True,
    )

    missing_timestamps = int(
        parsed_dates.isna().sum()
    )

    if missing_timestamps:
        issues.append(
            _issue(
                "INVALID_OR_MISSING_TIMESTAMPS",
                "high",
                "validity_consistency",
                "Some timestamps are missing or cannot be parsed.",
                column=datetime_column,
                recommendation=(
                    "Correct missing and invalid timestamps "
                    "before constructing chronological splits."
                ),
                evidence={
                    "count": missing_timestamps,
                    "percentage": _percentage(
                        missing_timestamps,
                        total_rows,
                    ),
                },
            )
        )

    # Evaluate every series separately. dropna=False ensures
    # rows with a missing group identifier are not discarded.
    series_results: list[dict[str, Any]] = []

    if group_column:
        grouped = dataframe.groupby(
            group_column,
            dropna=False,
            sort=False,
        )

        for label, group in grouped:
            series_dates = parsed_dates.loc[group.index]

            series_result, series_issues = _analyze_series(
                group,
                series_dates,
                target_column=target_column,
                datetime_column=datetime_column,
                series_label=str(label),
            )

            series_results.append(series_result)
            issues.extend(series_issues)

    else:
        series_result, series_issues = _analyze_series(
            dataframe,
            parsed_dates,
            target_column=target_column,
            datetime_column=datetime_column,
            series_label="default",
        )

        series_results.append(series_result)
        issues.extend(series_issues)

    if numeric_target:
        numeric = pd.to_numeric(
            dataframe[target_column],
            errors="coerce",
        ).astype(float)

        missing_target_count = int(
            numeric.isna().sum()
        )

        infinite_target_count = int(
            np.isinf(numeric.to_numpy()).sum()
        )

        finite_target = numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        unique_target_values = int(
            finite_target.nunique()
        )

        if missing_target_count:
            issues.append(
                _issue(
                    "MISSING_FORECAST_TARGET",
                    "high",
                    "target_suitability",
                    "The forecasting target contains missing values.",
                    column=target_column,
                    evidence={
                        "count": missing_target_count,
                    },
                )
            )

        if infinite_target_count:
            issues.append(
                _issue(
                    "INFINITE_FORECAST_TARGET",
                    "high",
                    "target_suitability",
                    "The forecasting target contains infinite values.",
                    column=target_column,
                    evidence={
                        "count": infinite_target_count,
                    },
                )
            )

        if unique_target_values <= 1:
            issues.append(
                _issue(
                    "CONSTANT_FORECAST_TARGET",
                    "critical",
                    "target_suitability",
                    "The target has insufficient observed variation.",
                    column=target_column,
                )
            )

        target_score = round(
            100.0 * len(finite_target) / total_rows,
            2,
        )

        if unique_target_values <= 1:
            target_score = 0.0

    else:
        missing_target_count = None
        infinite_target_count = None
        unique_target_values = None
        target_score = 0.0

    # The shortest series limits the available history
    # when multiple independent series are present.
    minimum_series_rows = min(
        result["rows"]
        for result in series_results
    )

    history_score = round(
        min(
            100.0,
            100.0 * minimum_series_rows / MIN_HISTORY_ROWS,
        ),
        2,
    )

    if minimum_series_rows < MIN_HISTORY_ROWS:
        issues.append(
            _issue(
                "SHORT_FORECAST_HISTORY",
                "medium",
                "data_sufficiency",
                "At least one series has fewer than 100 observations.",
                recommendation=(
                    "Review the historical coverage required "
                    "for the forecasting horizon and seasonality."
                ),
                evidence={
                    "shortest_series_rows": minimum_series_rows,
                    "initial_threshold": MIN_HISTORY_ROWS,
                },
            )
        )

    # Estimate a simple chronological holdout.
    validation_rows = math.ceil(
        minimum_series_rows * validation_fraction
    )

    training_rows = (
        minimum_series_rows - validation_rows
    )

    holdout_feasible = (
        training_rows >= MIN_TRAIN_ROWS
        and validation_rows >= MIN_VALIDATION_ROWS
    )

    suggested_strategy = (
        "time_series_split"
        if validation_strategy == "auto"
        else validation_strategy
    )

    if suggested_strategy in {
        "random_split",
        "stratified_split",
    }:
        issues.append(
            _issue(
                "NON_CHRONOLOGICAL_VALIDATION",
                "high",
                "leakage_validation",
                (
                    "Random or stratified splitting may train "
                    "on observations that occur after validation data."
                ),
                recommendation=(
                    "Use chronological validation for forecasting "
                    "unless a different design is explicitly justified."
                ),
            )
        )

    if not holdout_feasible:
        issues.append(
            _issue(
                "INSUFFICIENT_FORECAST_HOLDOUT",
                "high",
                "data_sufficiency",
                (
                    "The shortest series has insufficient history "
                    "for the preliminary holdout thresholds."
                ),
                evidence={
                    "estimated_training_rows": training_rows,
                    "estimated_validation_rows": validation_rows,
                },
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
        "task": "forecasting",
        "target_column": target_column,
        "datetime_column": datetime_column,
        "group_column": group_column,
        "intended_algorithm": intended_algorithm,
        "total_rows": total_rows,
        "series_count": len(series_results),
        "series": series_results,
        "missing_timestamp_count": missing_timestamps,
        "missing_target_count": missing_target_count,
        "infinite_target_count": infinite_target_count,
        "unique_target_values": unique_target_values,
        "scores": {
            "target_suitability": target_score,
            "data_sufficiency": history_score,
        },
        "validation": {
            "requested_strategy": validation_strategy,
            "suggested_strategy": suggested_strategy,
            "validation_fraction": validation_fraction,
            "estimated_training_rows_per_shortest_series": (
                training_rows
            ),
            "estimated_validation_rows_per_shortest_series": (
                validation_rows
            ),
            "holdout_feasible": holdout_feasible,
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
                "The inferred observation frequency is only "
                "a candidate; business calendars and seasonality "
                "require domain-specific configuration."
            ),
            (
                "History thresholds are preliminary and do not "
                "account for the requested forecasting horizon."
            ),
            (
                "Actual chronological splits, temporal gaps and "
                "prediction-time feature availability are not "
                "verified by this module."
            ),
        ],
    }
