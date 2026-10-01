
"""NUMPA v2 - Data leakage and validation-risk detection.

Detects structural leakage problems and statistical warning signs.

Important:
- High correlation does not prove leakage.
- Identifier-like features are warnings, not automatic exclusions.
- Split checks require actual train/validation row positions.
- Temporal leakage also depends on when features become available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas.readiness import (
    MLTask,
    ReadinessIssue,
)


# Configurable preliminary thresholds.
HIGH_TARGET_CORRELATION = 0.995
IDENTIFIER_UNIQUENESS_RATIO = 0.98
MIN_CORRELATION_ROWS = 10


def _make_issue(
    code: str,
    severity: str,
    message: str,
    *,
    column: str | None = None,
    recommendation: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> ReadinessIssue:
    """Create an issue using the shared readiness schema."""

    return ReadinessIssue(
        code=code,
        severity=severity,
        dimension="leakage_validation",
        message=message,
        column=column,
        recommendation=recommendation,
        evidence=evidence or {},
    )


def _validate_positions(
    positions: list[int],
    row_count: int,
    name: str,
) -> list[int]:
    """Validate positional train/validation row selections."""

    if not positions:
        raise ValueError(f"{name} cannot be empty.")

    if len(positions) != len(set(positions)):
        raise ValueError(f"{name} contains duplicate row positions.")

    for position in positions:
        if isinstance(position, bool) or not isinstance(
            position, (int, np.integer)
        ):
            raise ValueError(
                f"{name} must contain integer row positions."
            )

        if position < 0 or position >= row_count:
            raise ValueError(
                f"{name} contains an out-of-range row position."
            )

    return [int(position) for position in positions]


def _identical_observed_values(
    feature: pd.Series,
    target: pd.Series,
) -> bool:
    """Check whether two columns contain the same observed values.

    Missing-value locations must also match. Comparison uses
    pandas equality and does not infer causal relationships.
    """

    if not feature.isna().equals(target.isna()):
        return False

    # String conversion is deliberately avoided so different
    # data types do not accidentally become equivalent.
    try:
        equal = feature.eq(target)
        return bool(equal.fillna(True).all())
    except (TypeError, ValueError):
        return False


def detect_target_leakage(
    dataframe: pd.DataFrame,
    *,
    task: MLTask,
    target_column: str | None,
    excluded_columns: list[str] | None = None,
    datetime_column: str | None = None,
    group_column: str | None = None,
) -> list[ReadinessIssue]:
    """Inspect input features for target-related leakage risks."""

    if task == "clustering":
        return []

    if not target_column or target_column not in dataframe.columns:
        raise ValueError(
            "A valid target column is required for supervised tasks."
        )

    excluded = set(excluded_columns or [])

    # Role columns should not be treated as ordinary features
    # unless the modeling workflow explicitly includes them.
    excluded.add(target_column)

    if datetime_column:
        excluded.add(datetime_column)

    if group_column:
        excluded.add(group_column)

    target = dataframe[target_column]
    issues: list[ReadinessIssue] = []

    for column in dataframe.columns:
        if column in excluded:
            continue

        feature = dataframe[column]

        # Direct duplication of the target is a structural issue.
        if _identical_observed_values(feature, target):
            issues.append(
                _make_issue(
                    "TARGET_DUPLICATED_IN_FEATURE",
                    "critical",
                    "This feature contains the same values as the target.",
                    column=str(column),
                    recommendation=(
                        "Remove the duplicated target feature from "
                        "model inputs and review how it was created."
                    ),
                    evidence={
                        "target_column": target_column,
                        "observed_rows": int(target.notna().sum()),
                    },
                )
            )

            # Avoid reporting a second, weaker correlation
            # warning for the same duplicated target.
            continue

        # Correlation is only meaningful for numeric features
        # and numeric targets with sufficient paired observations.
        if (
            pd.api.types.is_numeric_dtype(feature)
            and pd.api.types.is_numeric_dtype(target)
        ):
            paired = pd.DataFrame({
                "feature": pd.to_numeric(
                    feature,
                    errors="coerce",
                ),
                "target": pd.to_numeric(
                    target,
                    errors="coerce",
                ),
            })

            # Exclude missing and infinite values.
            paired = paired.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if (
                len(paired) >= MIN_CORRELATION_ROWS
                and paired["feature"].nunique() > 1
                and paired["target"].nunique() > 1
            ):
                correlation = paired["feature"].corr(
                    paired["target"]
                )

                if (
                    pd.notna(correlation)
                    and abs(float(correlation))
                    >= HIGH_TARGET_CORRELATION
                ):
                    issues.append(
                        _make_issue(
                            "HIGH_TARGET_CORRELATION",
                            "medium",
                            (
                                "This feature has an unusually strong "
                                "linear relationship with the target."
                            ),
                            column=str(column),
                            recommendation=(
                                "Verify that the feature is available "
                                "at prediction time and is not derived "
                                "from the target or future outcomes."
                            ),
                            evidence={
                                "absolute_correlation": round(
                                    abs(float(correlation)),
                                    6,
                                ),
                                "paired_rows": len(paired),
                                "threshold": HIGH_TARGET_CORRELATION,
                            },
                        )
                    )

    return issues


def detect_identifier_risks(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    excluded_columns: list[str] | None = None,
) -> list[ReadinessIssue]:
    """Flag likely identifier columns that could cause memorization.

    High uniqueness alone is not enough to flag a numeric
    measurement. Column naming also provides supporting evidence.
    """

    excluded = set(excluded_columns or [])

    if target_column:
        excluded.add(target_column)

    issues: list[ReadinessIssue] = []

    for column in dataframe.columns:
        if column in excluded:
            continue

        name = str(column).strip().lower()

        identifier_name = (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or name in {
                "uuid",
                "guid",
                "record_id",
                "customer_id",
                "user_id",
                "transaction_id",
            }
        )

        if not identifier_name:
            continue

        series = dataframe[column].dropna()

        if series.empty:
            continue

        uniqueness_ratio = (
            float(series.nunique()) / len(series)
        )

        if uniqueness_ratio >= IDENTIFIER_UNIQUENESS_RATIO:
            issues.append(
                _make_issue(
                    "IDENTIFIER_LIKE_FEATURE",
                    "medium",
                    (
                        "This column resembles a nearly unique "
                        "record identifier."
                    ),
                    column=str(column),
                    recommendation=(
                        "Review whether this identifier should be "
                        "excluded from model features or used for "
                        "group-aware validation."
                    ),
                    evidence={
                        "uniqueness_ratio": round(
                            uniqueness_ratio,
                            6,
                        ),
                        "threshold": IDENTIFIER_UNIQUENESS_RATIO,
                    },
                )
            )

    return issues


def detect_split_leakage(
    dataframe: pd.DataFrame,
    *,
    train_positions: list[int],
    validation_positions: list[int],
    feature_columns: list[str] | None = None,
    group_column: str | None = None,
    datetime_column: str | None = None,
) -> list[ReadinessIssue]:
    """Inspect an existing train/validation split for leakage.

    Positions are zero-based row positions, not DataFrame
    index labels. This avoids ambiguity with duplicate indexes.
    """

    row_count = len(dataframe)

    train = _validate_positions(
        train_positions,
        row_count,
        "train_positions",
    )

    validation = _validate_positions(
        validation_positions,
        row_count,
        "validation_positions",
    )

    issues: list[ReadinessIssue] = []

    # A row appearing in both sets is definite split overlap.
    overlap = set(train).intersection(validation)

    if overlap:
        issues.append(
            _make_issue(
                "TRAIN_VALIDATION_ROW_OVERLAP",
                "critical",
                "The same row positions occur in both data splits.",
                recommendation=(
                    "Create disjoint training and validation sets."
                ),
                evidence={
                    "overlapping_row_count": len(overlap),
                },
            )
        )

    train_df = dataframe.iloc[train]
    validation_df = dataframe.iloc[validation]

    # Group-aware validation should keep each group in
    # exactly one split.
    if group_column:
        if group_column not in dataframe.columns:
            raise ValueError(
                f"Group column not found: {group_column}"
            )

        train_groups = set(
            train_df[group_column].dropna().tolist()
        )
        validation_groups = set(
            validation_df[group_column].dropna().tolist()
        )

        shared_groups = train_groups.intersection(
            validation_groups
        )

        if shared_groups:
            issues.append(
                _make_issue(
                    "GROUP_OVERLAP",
                    "high",
                    (
                        "Some groups appear in both training "
                        "and validation sets."
                    ),
                    column=group_column,
                    recommendation=(
                        "Use a group-aware split when observations "
                        "from the same group are related."
                    ),
                    evidence={
                        "shared_group_count": len(shared_groups),
                    },
                )
            )

    # Chronological validation must train only on earlier data.
    if datetime_column:
        if datetime_column not in dataframe.columns:
            raise ValueError(
                f"Datetime column not found: {datetime_column}"
            )

        dates = pd.to_datetime(
            dataframe[datetime_column],
            errors="coerce",
            utc=True,
        )

        train_dates = dates.iloc[train].dropna()
        validation_dates = dates.iloc[validation].dropna()

        if not train_dates.empty and not validation_dates.empty:
            latest_train = train_dates.max()
            earliest_validation = validation_dates.min()

            if latest_train >= earliest_validation:
                issues.append(
                    _make_issue(
                        "TEMPORAL_SPLIT_OVERLAP",
                        "high",
                        (
                            "Training timestamps overlap with or "
                            "follow validation timestamps."
                        ),
                        column=datetime_column,
                        recommendation=(
                            "Use chronological splitting and apply "
                            "an appropriate temporal gap if needed."
                        ),
                        evidence={
                            "latest_training_timestamp": (
                                latest_train.isoformat()
                            ),
                            "earliest_validation_timestamp": (
                                earliest_validation.isoformat()
                            ),
                        },
                    )
                )

    # Detect identical feature records crossing split boundaries.
    # This is a warning: legitimate repeated observations may
    # exist, depending on the dataset and modeling objective.
    if feature_columns is None:
        excluded = {
            column
            for column in (group_column, datetime_column)
            if column is not None
        }

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in excluded
        ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unknown feature columns: {missing_columns}"
        )

    if feature_columns:
        # Hash complete feature rows without depending on the
        # DataFrame's index labels.
        try:
            train_hashes = pd.util.hash_pandas_object(
                train_df[feature_columns],
                index=False,
            )

            validation_hashes = pd.util.hash_pandas_object(
                validation_df[feature_columns],
                index=False,
            )

            shared_hashes = set(
                train_hashes.tolist()
            ).intersection(
                validation_hashes.tolist()
            )

            duplicate_validation_rows = int(
                validation_hashes.isin(shared_hashes).sum()
            )

        except (TypeError, ValueError):
            # Nested or unhashable feature values can prevent
            # this optional check.
            duplicate_validation_rows = 0

        if duplicate_validation_rows:
            issues.append(
                _make_issue(
                    "DUPLICATE_FEATURE_ROWS_ACROSS_SPLITS",
                    "medium",
                    (
                        "Identical feature records appear in both "
                        "training and validation sets."
                    ),
                    recommendation=(
                        "Investigate repeated observations and "
                        "choose a split strategy appropriate to "
                        "the prediction task."
                    ),
                    evidence={
                        "matching_validation_rows": (
                            duplicate_validation_rows
                        ),
                    },
                )
            )

    return issues


def analyze_leakage(
    dataframe: pd.DataFrame,
    *,
    task: MLTask,
    target_column: str | None = None,
    excluded_columns: list[str] | None = None,
    group_column: str | None = None,
    datetime_column: str | None = None,
    train_positions: list[int] | None = None,
    validation_positions: list[int] | None = None,
) -> dict[str, Any]:
    """Run available leakage checks and return an auditable report."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("A pandas DataFrame is required.")

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if not dataframe.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")

    if task not in (
        "classification",
        "regression",
        "forecasting",
        "clustering",
    ):
        raise ValueError(f"Unsupported ML task: {task}")

    excluded = list(excluded_columns or [])

    issues = detect_target_leakage(
        dataframe,
        task=task,
        target_column=target_column,
        excluded_columns=excluded,
        datetime_column=datetime_column,
        group_column=group_column,
    )

    issues.extend(
        detect_identifier_risks(
            dataframe,
            target_column=target_column,
            excluded_columns=excluded,
        )
    )

    split_checked = False

    # Both split lists must be supplied together.
    if (train_positions is None) != (
        validation_positions is None
    ):
        raise ValueError(
            "Provide both training and validation row positions."
        )

    if (
        train_positions is not None
        and validation_positions is not None
    ):
        feature_exclusions = set(excluded)

        if target_column:
            feature_exclusions.add(target_column)

        if group_column:
            feature_exclusions.add(group_column)

        if datetime_column:
            feature_exclusions.add(datetime_column)

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in feature_exclusions
        ]

        issues.extend(
            detect_split_leakage(
                dataframe,
                train_positions=train_positions,
                validation_positions=validation_positions,
                feature_columns=feature_columns,
                group_column=group_column,
                datetime_column=datetime_column,
            )
        )

        split_checked = True

    # Report counts rather than manufacturing an overall
    # readiness score from incomplete leakage coverage.
    critical_count = sum(
        issue.severity == "critical"
        for issue in issues
    )

    high_count = sum(
        issue.severity == "high"
        for issue in issues
    )

    return {
        "task": task,
        "target_column": target_column,
        "split_checked": split_checked,
        "issue_count": len(issues),
        "critical_issue_count": critical_count,
        "high_issue_count": high_count,
        "issues": [
            issue.model_dump(mode="json")
            for issue in issues
        ],
        "limitations": [
            (
                "Statistical relationships alone do not "
                "establish data leakage."
            ),
            (
                "Feature availability at prediction time "
                "requires domain-specific review."
            ),
            (
                "Preprocessing leakage cannot be fully assessed "
                "without inspecting fitted pipeline artifacts "
                "and the training workflow."
            ),
        ],
    }
