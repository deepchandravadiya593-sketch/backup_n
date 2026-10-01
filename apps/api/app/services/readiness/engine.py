
"""NUMPA v2 - Task-aware ML Readiness Engine.

Combines shared dataset diagnostics with classification,
regression, forecasting or clustering-specific checks.

This implementation produces a provisional assessment only.
Leakage and validation checks remain incomplete, so the
overall final readiness score is deliberately withheld.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from app.schemas.readiness import (
    DEFAULT_DIMENSION_WEIGHTS,
    ReadinessAssessmentRequest,
    ReadinessConfiguration,
    ReadinessDimensionResult,
    ReadinessIssue,
)
from app.services.readiness.dimensions import evaluate_dimensions
from app.services.readiness.leakage import analyze_leakage
from app.services.readiness.tasks.classification import (
    assess_classification,
)
from app.services.readiness.tasks.regression import (
    assess_regression,
)
from app.services.readiness.tasks.forecasting import (
    assess_forecasting,
)
from app.services.readiness.tasks.clustering import (
    assess_clustering,
)


ENGINE_VERSION = "0.2.0-task-aware"

KNOWN_DIMENSIONS = frozenset(
    DEFAULT_DIMENSION_WEIGHTS
)

# Critical issues block a clean readiness result.
BLOCKING_LEAKAGE_CODES = {
    "TARGET_DUPLICATED_IN_FEATURE",
    "TRAIN_VALIDATION_ROW_OVERLAP",
}

# Task-specific modules currently provide preliminary
# replacements for these shared dimension scores.
TASK_SCORE_DIMENSIONS = {
    "classification": {
        "target_suitability",
        "data_sufficiency",
    },
    "regression": {
        "target_suitability",
        "data_sufficiency",
    },
    "forecasting": {
        "target_suitability",
        "data_sufficiency",
    },
    "clustering": {
        "feature_suitability",
        "data_sufficiency",
    },
}


def _utc_now() -> str:
    """Return a timezone-aware assessment timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _resolve_configuration(
    request: ReadinessAssessmentRequest,
) -> ReadinessConfiguration:
    """Resolve and validate the selected scoring profile."""

    configuration = (
        request.configuration.model_copy(deep=True)
        if request.configuration is not None
        else ReadinessConfiguration()
    )

    # Revalidate the model because Pydantic models may have
    # been modified after their initial construction.
    configuration = ReadinessConfiguration.model_validate(
        configuration.model_dump(mode="python")
    )

    unknown = (
        set(configuration.dimensions)
        - KNOWN_DIMENSIONS
    )

    if unknown:
        raise ValueError(
            "Unknown readiness dimensions: "
            + ", ".join(sorted(unknown))
        )

    # Custom thresholds and parameters are intentionally
    # rejected until the corresponding scoring logic exists.
    for name, settings in configuration.dimensions.items():
        if settings.thresholds:
            raise ValueError(
                f"Custom thresholds for '{name}' "
                "are not implemented yet."
            )

    if configuration.parameters:
        raise ValueError(
            "Custom scoring parameters are not implemented yet."
        )

    return configuration


def _validate_columns(
    dataframe: pd.DataFrame,
    request: ReadinessAssessmentRequest,
) -> None:
    """Validate dataset shape and referenced column roles."""

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

    referenced = [
        request.target_column,
        request.datetime_column,
        request.group_column,
        *request.excluded_columns,
    ]

    unknown = [
        column
        for column in referenced
        if (
            column is not None
            and column not in dataframe.columns
        )
    ]

    if unknown:
        raise ValueError(
            "Referenced columns do not exist: "
            + ", ".join(sorted(set(unknown)))
        )

    if (
        request.target_column is not None
        and request.target_column
        in {
            request.datetime_column,
            request.group_column,
        }
    ):
        raise ValueError(
            "The target column must be distinct from "
            "the datetime and group columns."
        )


def _normalize_pending(
    pending: list[Any],
) -> list[dict[str, str]]:
    """Normalize pending dimensions from older engine formats.

    File 30 may return a list of dimension names, while
    newer services may return dictionaries.
    """

    normalized: list[dict[str, str]] = []

    for item in pending:
        if isinstance(item, str):
            normalized.append({
                "dimension": item,
                "reason": "Dimension assessment is incomplete.",
            })

        elif isinstance(item, dict):
            name = item.get("dimension")

            if not isinstance(name, str):
                raise ValueError(
                    "A pending dimension has no valid name."
                )

            normalized.append({
                "dimension": name,
                "reason": str(
                    item.get(
                        "reason",
                        "Dimension assessment is incomplete.",
                    )
                ),
            })

        else:
            raise ValueError(
                "Unsupported pending dimension format."
            )

    return normalized


def _run_task_checks(
    dataframe: pd.DataFrame,
    request: ReadinessAssessmentRequest,
) -> dict[str, Any]:
    """Execute exactly one task-specific diagnostic module."""

    if request.task == "classification":
        return assess_classification(
            dataframe,
            target_column=request.target_column,
            validation_strategy=request.validation_strategy,
            intended_algorithm=request.intended_algorithm,
        )

    if request.task == "regression":
        return assess_regression(
            dataframe,
            target_column=request.target_column,
            validation_strategy=request.validation_strategy,
            intended_algorithm=request.intended_algorithm,
        )

    if request.task == "forecasting":
        return assess_forecasting(
            dataframe,
            target_column=request.target_column,
            datetime_column=request.datetime_column,
            group_column=request.group_column,
            validation_strategy=request.validation_strategy,
            intended_algorithm=request.intended_algorithm,
        )

    if request.task == "clustering":
        return assess_clustering(
            dataframe,
            excluded_columns=request.excluded_columns,
            intended_algorithm=request.intended_algorithm,
        )

    raise ValueError(
        f"Unsupported ML task: {request.task}"
    )


def _merge_issues(
    *issue_groups: list[ReadinessIssue],
) -> list[ReadinessIssue]:
    """Deduplicate issues while retaining their evidence.

    Issue codes alone are not sufficient because the same
    problem can occur in several different columns.
    """

    merged: list[ReadinessIssue] = []
    seen: set[str] = set()

    for group in issue_groups:
        for issue in group:
            # Serialize all fields to preserve distinct
            # column-specific and series-specific evidence.
            key = issue.model_dump_json(
                exclude_none=True
            )

            if key not in seen:
                seen.add(key)
                merged.append(issue)

    return merged


def _apply_configuration(
    dimension: ReadinessDimensionResult,
    configuration: ReadinessConfiguration,
) -> ReadinessDimensionResult | None:
    """Apply the approved weight to an assessed dimension."""

    settings = configuration.dimensions.get(
        dimension.name
    )

    if settings is None or not settings.enabled:
        return None

    updated = dimension.model_copy(deep=True)

    updated.weight = settings.weight
    updated.weighted_score = round(
        updated.score * settings.weight / 100.0,
        4,
    )

    return updated


def _replace_task_scores(
    dimensions: list[ReadinessDimensionResult],
    task_report: dict[str, Any],
    *,
    task: str,
) -> list[ReadinessDimensionResult]:
    """Replace preliminary shared scores with task-specific scores.

    The shared dimensions supply the explanation, while
    the task module supplies its more specific measurement.
    """

    replacements = TASK_SCORE_DIMENSIONS[task]
    task_scores = task_report.get("scores", {})

    updated: list[ReadinessDimensionResult] = []

    for original in dimensions:
        dimension = original.model_copy(deep=True)

        if (
            dimension.name in replacements
            and dimension.name in task_scores
        ):
            score = float(
                task_scores[dimension.name]
            )

            if not 0.0 <= score <= 100.0:
                raise ValueError(
                    "Task-specific score is outside 0–100: "
                    f"{dimension.name}"
                )

            dimension.score = round(score, 2)

            dimension.weighted_score = round(
                dimension.score
                * dimension.weight
                / 100.0,
                4,
            )

            dimension.status = (
                "ready"
                if dimension.score >= 80
                else "needs_attention"
                if dimension.score >= 50
                else "not_ready"
            )

            dimension.explanation = (
                f"Preliminary {task}-specific assessment. "
                "Review the task diagnostics for the "
                "underlying measurements and limitations."
            )

            dimension.metrics = {
                **dimension.metrics,
                "task_specific": True,
                "task": task,
            }

            # The task report's issues are included in the
            # combined assessment separately below.
            dimension.issues = []

        updated.append(dimension)

    return updated


def _make_clustering_target_dimension(
    task_report: dict[str, Any],
) -> ReadinessDimensionResult:
    """Interpret target suitability for unsupervised clustering.

    A clustering target is not required. Its default weight
    is retained temporarily, but this is a provisional
    convention rather than a finalized clustering formula.
    """

    weight = DEFAULT_DIMENSION_WEIGHTS[
        "target_suitability"
    ]

    return ReadinessDimensionResult(
        name="target_suitability",
        score=100.0,
        weight=weight,
        weighted_score=weight,
        status="not_applicable",
        explanation=(
            "Clustering is unsupervised and does not "
            "require a target column. This dimension "
            "needs an approved clustering-specific "
            "weight redistribution before final scoring."
        ),
        issues=[],
        metrics={
            "task": "clustering",
            "applicable": False,
        },
    )


def _leakage_diagnostics(
    report: dict[str, Any],
    *,
    split_supplied: bool,
) -> dict[str, Any]:
    """Represent leakage findings without inventing a score."""

    issues = [
        ReadinessIssue.model_validate(item)
        for item in report.get("issues", [])
    ]

    blocking = [
        issue
        for issue in issues
        if (
            issue.severity == "critical"
            or issue.code in BLOCKING_LEAKAGE_CODES
        )
    ]

    pending_checks = [
        "Prediction-time feature availability has not been verified.",
        "Preprocessing and feature-selection boundaries have not been audited.",
    ]

    if not split_supplied:
        pending_checks.append(
            "Actual train/validation row assignments have not been checked."
        )

    return {
        "dimension": "leakage_validation",
        "status": "pending",
        "score": None,
        "issues": [
            issue.model_dump(mode="json")
            for issue in issues
        ],
        "blocking_issues": [
            issue.model_dump(mode="json")
            for issue in blocking
        ],
        "pending_checks": pending_checks,
    }


def assess_readiness(
    dataframe: pd.DataFrame,
    request: ReadinessAssessmentRequest,
    *,
    train_positions: list[int] | None = None,
    validation_positions: list[int] | None = None,
) -> dict[str, Any]:
    """Create a task-aware provisional ML readiness assessment.

    Never return a final overall score while required
    scoring dimensions or leakage checks are pending.
    """

    _validate_columns(
        dataframe,
        request,
    )

    configuration = _resolve_configuration(
        request
    )

    if (
        (train_positions is None)
        != (validation_positions is None)
    ):
        raise ValueError(
            "Both train_positions and validation_positions "
            "must be provided together."
        )

    split_supplied = (
        train_positions is not None
        and validation_positions is not None
    )

    # Step 1: Shared, task-aware preliminary dimensions.
    preliminary = evaluate_dimensions(
        dataframe,
        task=request.task,
        target_column=request.target_column,
        excluded_columns=request.excluded_columns,
    )

    dimensions = [
        ReadinessDimensionResult.model_validate(item)
        for item in preliminary["dimensions"]
    ]

    pending = _normalize_pending(
        preliminary.get("pending_dimensions", [])
    )

    # Step 2: Run the relevant specialized task module.
    task_report = _run_task_checks(
        dataframe,
        request,
    )

    dimensions = _replace_task_scores(
        dimensions,
        task_report,
        task=request.task,
    )

    # Step 3: Clustering's target dimension is not applicable.
    # Do not assign a misleading score or count its weight
    # in the provisional measured coverage.
    if request.task == "clustering":
        pending = [
            item
            for item in pending
            if item["dimension"] != "target_suitability"
        ]

        pending.append({
            "dimension": "target_suitability",
            "reason": (
                "Clustering requires an approved redistribution "
                "of the target-suitability weight."
            ),
        })

    # Step 4: Collect shared and task-specific issues.
    shared_issues = [
        issue
        for dimension in dimensions
        for issue in dimension.issues
    ]

    task_issues = [
        ReadinessIssue.model_validate(item)
        for item in task_report.get("issues", [])
    ]

    # Step 5: Run preliminary leakage diagnostics.
    leakage_report = analyze_leakage(
        dataframe,
        task=request.task,
        target_column=request.target_column,
        excluded_columns=request.excluded_columns,
        group_column=request.group_column,
        datetime_column=request.datetime_column,
        train_positions=train_positions,
        validation_positions=validation_positions,
    )

    leakage = _leakage_diagnostics(
        leakage_report,
        split_supplied=split_supplied,
    )

    leakage_issues = [
        ReadinessIssue.model_validate(item)
        for item in leakage["issues"]
    ]

    all_issues = _merge_issues(
        shared_issues,
        task_issues,
        leakage_issues,
    )

    blocking_issues = [
        issue
        for issue in all_issues
        if (
            issue.severity == "critical"
            or issue.code in BLOCKING_LEAKAGE_CODES
        )
    ]

    # Step 6: Apply configured weights only to dimensions
    # whose preliminary scores have been calculated.
    measured_dimensions: list[
        ReadinessDimensionResult
    ] = []

    for dimension in dimensions:
        configured = _apply_configuration(
            dimension,
            configuration,
        )

        if configured is not None:
            measured_dimensions.append(configured)

    # Explicitly track all enabled dimensions that have
    # not yet received a complete measurement.
    measured_names = {
        dimension.name
        for dimension in measured_dimensions
    }

    enabled_names = {
        name
        for name, settings
        in configuration.dimensions.items()
        if settings.enabled
    }

    for name in sorted(
        enabled_names - measured_names
    ):
        if not any(
            item["dimension"] == name
            for item in pending
        ):
            pending.append({
                "dimension": name,
                "reason": (
                    "The enabled dimension has not "
                    "been fully assessed."
                ),
            })

    # Leakage remains pending even if a previous shared
    # module attempted to include a preliminary score.
    measured_dimensions = [
        dimension
        for dimension in measured_dimensions
        if dimension.name != "leakage_validation"
    ]

    pending = [
        item
        for item in pending
        if item["dimension"] != "leakage_validation"
    ]

    if (
        configuration.dimensions.get(
            "leakage_validation"
        ) is not None
        and configuration.dimensions[
            "leakage_validation"
        ].enabled
    ):
        pending.append({
            "dimension": "leakage_validation",
            "reason": (
                "Leakage and validation checks "
                "remain incomplete."
            ),
        })

    # Exclude clustering target suitability from measured
    # coverage until its weight is redistributed.
    if request.task == "clustering":
        measured_dimensions = [
            dimension
            for dimension in measured_dimensions
            if dimension.name != "target_suitability"
        ]

    measured_weight = round(
        sum(
            dimension.weight
            for dimension in measured_dimensions
        ),
        4,
    )

    provisional_score = (
        round(
            sum(
                dimension.weighted_score
                for dimension in measured_dimensions
            )
            * 100.0
            / measured_weight,
            2,
        )
        if measured_weight > 0
        else None
    )

    # Deduplicate pending dimensions by name.
    unique_pending: dict[str, dict[str, str]] = {}

    for item in pending:
        unique_pending[
            item["dimension"]
        ] = item

    pending = list(
        unique_pending.values()
    )

    recommendations = list(
        dict.fromkeys(
            issue.recommendation
            for issue in all_issues
            if issue.recommendation
        )
    )

    return {
        "assessment_id": str(uuid4()),
        "dataset_id": request.dataset_id,
        "task": request.task,
        "target_column": request.target_column,
        "intended_algorithm": request.intended_algorithm,
        "validation_strategy": request.validation_strategy,
        "configuration": {
            "configuration_id": (
                configuration.configuration_id
            ),
            "name": configuration.name,
            "version": configuration.version,
        },
        "assessment_status": "provisional",
        "overall_score": None,
        "provisional_score": provisional_score,
        "measured_weight_percentage": measured_weight,
        "pending_weight_percentage": round(
            max(0.0, 100.0 - measured_weight),
            4,
        ),
        "dimensions": [
            dimension.model_dump(mode="json")
            for dimension in measured_dimensions
        ],
        "pending_dimensions": pending,
        "task_diagnostics": task_report,
        "leakage": leakage,
        "issues": [
            issue.model_dump(mode="json")
            for issue in all_issues
        ],
        "blocking_issues": [
            issue.model_dump(mode="json")
            for issue in blocking_issues
        ],
        "recommendations": recommendations,
        "assessed_at": _utc_now(),
        "audit": {
            "engine_version": ENGINE_VERSION,
            "dataset_rows": int(len(dataframe)),
            "dataset_columns": int(
                len(dataframe.columns)
            ),
            "measured_dimensions": [
                dimension.name
                for dimension in measured_dimensions
            ],
            "pending_dimensions": [
                item["dimension"]
                for item in pending
            ],
            "actual_split_checked": split_supplied,
            "configuration_weights": {
                name: settings.weight
                for name, settings
                in configuration.dimensions.items()
                if settings.enabled
            },
            "limitations": [
                (
                    "Task-specific scores are preliminary "
                    "and use fixed diagnostic thresholds."
                ),
                (
                    "A final leakage/validation score "
                    "has not been calculated."
                ),
                (
                    "The final overall readiness score "
                    "is intentionally withheld."
                ),
            ],
        },
    }
