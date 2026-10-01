
"""NUMPA v2 - Clustering-specific ML readiness checks.

Provides deterministic diagnostics for unsupervised clustering.

Checks numeric feature compatibility, missing values, infinite
values, constant features, duplicates, scale differences and
preliminary sample sufficiency.

This module does not fit a clustering model or claim that
meaningful clusters exist in the dataset.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.readiness import ReadinessIssue


# Preliminary, configurable thresholds.
MIN_TOTAL_ROWS = 100
MIN_NUMERIC_FEATURES = 2
MAX_MISSING_PERCENTAGE = 50.0

# Warn when the largest nonzero feature standard deviation
# is at least 100 times the smallest.
SCALE_RATIO_THRESHOLD = 100.0

# Require at least two observations per intended cluster
# as a minimal feasibility heuristic.
MIN_ROWS_PER_CLUSTER = 2

# Avoid quadratic or unexpectedly expensive duplicate
# checks on larger datasets in this initial implementation.
MAX_DUPLICATE_CHECK_ROWS = 100_000


def _percentage(count: int, total: int) -> float:
    """Calculate a percentage safely."""

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
    """Build an issue using the shared readiness schema."""

    return ReadinessIssue(
        code=code,
        severity=severity,
        dimension=dimension,
        message=message,
        column=column,
        recommendation=recommendation,
        evidence=evidence or {},
    )


def _validate_cluster_count(
    intended_clusters: int | None,
) -> None:
    """Validate an optional intended number of clusters."""

    if intended_clusters is None:
        return

    if (
        isinstance(intended_clusters, bool)
        or not isinstance(
            intended_clusters,
            (int, np.integer),
        )
        or intended_clusters < 2
    ):
        raise ValueError(
            "intended_clusters must be an integer of at least 2."
        )


def _normalize_algorithm(
    intended_algorithm: str | None,
) -> str | None:
    """Normalize common algorithm names for diagnostic rules."""

    if intended_algorithm is None:
        return None

    normalized = (
        intended_algorithm
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "kmeans": "kmeans",
        "k_means": "kmeans",
        "mini_batch_kmeans": "minibatch_kmeans",
        "minibatch_kmeans": "minibatch_kmeans",
        "dbscan": "dbscan",
        "hdbscan": "hdbscan",
        "agglomerative": "hierarchical",
        "agglomerative_clustering": "hierarchical",
        "hierarchical": "hierarchical",
    }

    return aliases.get(normalized, normalized)


def _detect_duplicates(
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Check exact duplicate feature rows when feasible."""

    if len(features) > MAX_DUPLICATE_CHECK_ROWS:
        return {
            "checked": False,
            "count": None,
            "percentage": None,
            "reason": "Dataset exceeds the duplicate-check row limit.",
        }

    try:
        count = int(
            features.duplicated(keep="first").sum()
        )
    except (TypeError, ValueError):
        return {
            "checked": False,
            "count": None,
            "percentage": None,
            "reason": (
                "Some feature values cannot be compared "
                "for exact duplicates."
            ),
        }

    return {
        "checked": True,
        "count": count,
        "percentage": _percentage(count, len(features)),
        "reason": None,
    }


def assess_clustering(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: list[str] | None = None,
    intended_algorithm: str | None = None,
    intended_clusters: int | None = None,
) -> dict[str, Any]:
    """Generate a preliminary clustering-readiness report.

    Numeric-feature suitability and sample sufficiency are
    measured separately. Non-numeric columns are flagged
    because they may require appropriate encoding or a
    different distance measure.

    The report is diagnostic, not a final readiness score.
    """

    # Validate the dataset and requested configuration.
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("A pandas DataFrame is required.")

    if dataframe.empty:
        raise ValueError("Cannot assess an empty dataset.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Dataset column names must be unique."
        )

    _validate_cluster_count(intended_clusters)

    excluded = list(excluded_columns or [])

    if len(excluded) != len(set(excluded)):
        raise ValueError(
            "Excluded columns must not contain duplicates."
        )

    unknown = set(excluded) - set(dataframe.columns)

    if unknown:
        raise ValueError(
            "Unknown excluded columns: "
            + ", ".join(sorted(str(item) for item in unknown))
        )

    algorithm = _normalize_algorithm(intended_algorithm)

    # Never modify the original DataFrame.
    features = dataframe.drop(columns=excluded).copy(deep=True)

    if features.shape[1] == 0:
        raise ValueError(
            "No feature columns remain after exclusions."
        )

    row_count = int(len(features))
    feature_count = int(features.shape[1])

    issues: list[ReadinessIssue] = []

    numeric_columns = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
        and not pd.api.types.is_bool_dtype(features[column])
    ]

    non_numeric_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    # Inspect the numeric feature matrix.
    numeric = features[numeric_columns]

    numeric_missing_count = 0
    numeric_infinite_count = 0
    problematic_numeric_columns: set[str] = set()
    constant_numeric_columns: list[str] = []
    feature_statistics: list[dict[str, Any]] = []
    usable_standard_deviations: list[float] = []

    for column in numeric_columns:
        series = pd.to_numeric(
            numeric[column],
            errors="coerce",
        ).astype(float)

        values = series.to_numpy()

        missing_count = int(
            np.isnan(values).sum()
        )

        infinite_count = int(
            np.isinf(values).sum()
        )

        finite = series.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        missing_percentage = _percentage(
            missing_count,
            row_count,
        )

        numeric_missing_count += missing_count
        numeric_infinite_count += infinite_count

        if missing_count:
            issues.append(
                _issue(
                    "MISSING_CLUSTER_FEATURE",
                    (
                        "high"
                        if missing_percentage > MAX_MISSING_PERCENTAGE
                        else "medium"
                    ),
                    "completeness",
                    "A numeric feature contains missing values.",
                    column=str(column),
                    recommendation=(
                        "Review missing-value patterns and fit "
                        "any imputation method on training data "
                        "when evaluating generalization."
                    ),
                    evidence={
                        "missing_count": missing_count,
                        "missing_percentage": missing_percentage,
                    },
                )
            )

        if infinite_count:
            problematic_numeric_columns.add(str(column))

            issues.append(
                _issue(
                    "INFINITE_CLUSTER_FEATURE",
                    "high",
                    "validity_consistency",
                    "A numeric feature contains infinite values.",
                    column=str(column),
                    recommendation=(
                        "Investigate infinite values before "
                        "running a clustering algorithm."
                    ),
                    evidence={
                        "infinite_count": infinite_count,
                    },
                )
            )

        unique_count = int(finite.nunique())

        if unique_count <= 1:
            constant_numeric_columns.append(str(column))
            problematic_numeric_columns.add(str(column))

            issues.append(
                _issue(
                    "CONSTANT_CLUSTER_FEATURE",
                    "medium",
                    "feature_suitability",
                    "This numeric feature has no finite variation.",
                    column=str(column),
                    recommendation=(
                        "Review whether the feature provides "
                        "useful information for clustering."
                    ),
                    evidence={
                        "finite_unique_count": unique_count,
                    },
                )
            )

        if missing_percentage > MAX_MISSING_PERCENTAGE:
            problematic_numeric_columns.add(str(column))

        standard_deviation = (
            float(finite.std(ddof=0))
            if len(finite) > 1
            else None
        )

        if (
            standard_deviation is not None
            and math.isfinite(standard_deviation)
            and standard_deviation > 0
        ):
            usable_standard_deviations.append(
                standard_deviation
            )

        feature_statistics.append({
            "column": str(column),
            "missing_count": missing_count,
            "infinite_count": infinite_count,
            "finite_unique_count": unique_count,
            "standard_deviation": (
                round(standard_deviation, 6)
                if standard_deviation is not None
                else None
            ),
        })

    # The initial distance-based workflow requires numeric
    # features. Categorical and mixed-type clustering can
    # be supported later through suitable distance metrics.
    for column in non_numeric_columns:
        issues.append(
            _issue(
                "NON_NUMERIC_CLUSTER_FEATURE",
                "medium",
                "feature_suitability",
                "This feature is not numeric.",
                column=str(column),
                recommendation=(
                    "Choose an appropriate encoding or a "
                    "clustering algorithm supporting this "
                    "feature type."
                ),
                evidence={
                    "dtype": str(features[column].dtype),
                },
            )
        )

    if not numeric_columns:
        issues.append(
            _issue(
                "NO_NUMERIC_CLUSTER_FEATURES",
                "critical",
                "feature_suitability",
                (
                    "No numeric features are available for "
                    "the initial distance-based workflow."
                ),
                recommendation=(
                    "Select suitable numeric features or "
                    "explicitly encode categorical features."
                ),
            )
        )

    elif len(numeric_columns) < MIN_NUMERIC_FEATURES:
        issues.append(
            _issue(
                "FEW_NUMERIC_CLUSTER_FEATURES",
                "low",
                "feature_suitability",
                (
                    "Only one numeric feature is available. "
                    "One-dimensional clustering may still be valid."
                ),
            )
        )

    # Compare standard deviations only when at least two
    # varying numeric features are available.
    scale_ratio = None

    if len(usable_standard_deviations) >= 2:
        scale_ratio = (
            max(usable_standard_deviations)
            / min(usable_standard_deviations)
        )

        if scale_ratio >= SCALE_RATIO_THRESHOLD:
            issues.append(
                _issue(
                    "CLUSTER_FEATURE_SCALE_DIFFERENCE",
                    "medium",
                    "feature_suitability",
                    (
                        "Numeric features have substantially "
                        "different observed scales."
                    ),
                    recommendation=(
                        "Review feature scaling and whether "
                        "the differences reflect meaningful "
                        "domain-specific importance."
                    ),
                    evidence={
                        "standard_deviation_ratio": round(
                            scale_ratio,
                            4,
                        ),
                        "threshold": SCALE_RATIO_THRESHOLD,
                    },
                )
            )

    # Check duplicates using the actual selected features.
    duplicates = _detect_duplicates(features)

    if duplicates["checked"] and duplicates["count"] > 0:
        issues.append(
            _issue(
                "DUPLICATE_CLUSTER_OBSERVATIONS",
                "medium",
                "record_integrity",
                "Identical feature records occur more than once.",
                recommendation=(
                    "Determine whether repeated observations "
                    "represent meaningful frequency or "
                    "unintended duplicates."
                ),
                evidence={
                    "duplicate_count": duplicates["count"],
                    "duplicate_percentage": (
                        duplicates["percentage"]
                    ),
                },
            )
        )

    # Preliminary feature suitability:
    # count each numeric column at most once when it has
    # severe missingness, infinite values or no variation.
    suitable_numeric_count = (
        len(numeric_columns)
        - len(problematic_numeric_columns)
    )

    feature_suitability_score = round(
        100.0 * suitable_numeric_count / feature_count,
        2,
    )

    # Data sufficiency starts with a documented row-count
    # heuristic, then accounts for intended cluster count.
    data_sufficiency_score = min(
        100.0,
        100.0 * row_count / MIN_TOTAL_ROWS,
    )

    cluster_count_feasible = None

    if intended_clusters is not None:
        minimum_required_rows = (
            intended_clusters * MIN_ROWS_PER_CLUSTER
        )

        cluster_count_feasible = (
            row_count >= minimum_required_rows
            and intended_clusters <= row_count
        )

        if not cluster_count_feasible:
            issues.append(
                _issue(
                    "INSUFFICIENT_ROWS_FOR_CLUSTER_COUNT",
                    "high",
                    "data_sufficiency",
                    (
                        "The dataset has too few observations "
                        "for the intended number of clusters "
                        "under the preliminary heuristic."
                    ),
                    recommendation=(
                        "Review the intended cluster count "
                        "or obtain more observations."
                    ),
                    evidence={
                        "rows": row_count,
                        "intended_clusters": intended_clusters,
                        "minimum_required_rows": (
                            minimum_required_rows
                        ),
                    },
                )
            )

            data_sufficiency_score = min(
                data_sufficiency_score,
                100.0
                * row_count
                / minimum_required_rows,
            )

    if row_count < MIN_TOTAL_ROWS:
        issues.append(
            _issue(
                "LOW_CLUSTER_SAMPLE_COUNT",
                "medium",
                "data_sufficiency",
                "The dataset has fewer than 100 observations.",
                recommendation=(
                    "Review sample requirements for the "
                    "chosen clustering method and objective."
                ),
                evidence={
                    "rows": row_count,
                    "initial_threshold": MIN_TOTAL_ROWS,
                },
            )
        )

    # Algorithm-aware guidance is informational.
    algorithm_notes: list[str] = []

    if algorithm in {"kmeans", "minibatch_kmeans"}:
        algorithm_notes.append(
            "K-Means requires a numeric feature representation "
            "and is sensitive to feature scale and outliers."
        )

        if intended_clusters is None:
            algorithm_notes.append(
                "Specify or evaluate an appropriate number "
                "of clusters before fitting K-Means."
            )

    elif algorithm in {"dbscan", "hdbscan"}:
        algorithm_notes.append(
            "Density-based clustering is sensitive to "
            "distance metrics and feature scaling."
        )

        if intended_clusters is not None:
            algorithm_notes.append(
                "Density-based algorithms do not ordinarily "
                "require a fixed number of clusters."
            )

    elif algorithm == "hierarchical":
        algorithm_notes.append(
            "Hierarchical clustering can have substantial "
            "memory and computational costs on large datasets."
        )

    elif algorithm is not None:
        algorithm_notes.append(
            "No algorithm-specific checks are currently "
            "implemented for the selected algorithm."
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
        "task": "clustering",
        "intended_algorithm": algorithm,
        "intended_clusters": intended_clusters,
        "total_rows": row_count,
        "feature_count": feature_count,
        "numeric_feature_count": len(numeric_columns),
        "non_numeric_feature_count": len(non_numeric_columns),
        "numeric_columns": [
            str(column)
            for column in numeric_columns
        ],
        "non_numeric_columns": [
            str(column)
            for column in non_numeric_columns
        ],
        "feature_statistics": feature_statistics,
        "numeric_missing_count": numeric_missing_count,
        "numeric_infinite_count": numeric_infinite_count,
        "constant_numeric_columns": constant_numeric_columns,
        "scale_ratio": (
            round(scale_ratio, 4)
            if scale_ratio is not None
            else None
        ),
        "duplicates": duplicates,
        "scores": {
            "feature_suitability": (
                feature_suitability_score
            ),
            "data_sufficiency": round(
                max(
                    0.0,
                    min(100.0, data_sufficiency_score),
                ),
                2,
            ),
        },
        "cluster_count_feasible": cluster_count_feasible,
        "algorithm_notes": algorithm_notes,
        "issue_count": len(issues),
        "critical_issue_count": critical_count,
        "high_issue_count": high_count,
        "issues": [
            issue.model_dump(mode="json")
            for issue in issues
        ],
        "limitations": [
            (
                "These scores measure preliminary data "
                "suitability, not whether meaningful clusters exist."
            ),
            (
                "Numeric-feature checks assume an initial "
                "distance-based clustering workflow."
            ),
            (
                "No clustering model, silhouette score "
                "or cluster-stability analysis has been run."
            ),
            (
                "Scale differences and duplicate records "
                "require domain-specific interpretation."
            ),
        ],
    }
