
"""NUMPA v2 - Final scoring and validation rules.

Provides a deterministic gate for publishing a final
ML Readiness Score.

A final score is permitted only when:
- Every enabled scoring dimension is assessed.
- Effective dimension weights total 100.
- No critical blocking issues remain.
- Required task-specific validation is complete.
- The leakage audit is explicitly complete.
- Every assessed dimension has a valid numeric score.

This module does not perform validation or leakage checks.
It verifies evidence supplied by those services.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.readiness import (
    DEFAULT_DIMENSION_WEIGHTS,
    MLTask,
    ReadinessConfiguration,
    ReadinessDimensionResult,
    ReadinessIssue,
)


KNOWN_DIMENSIONS = frozenset(DEFAULT_DIMENSION_WEIGHTS)

FinalStatus = Literal[
    "ready",
    "needs_attention",
    "not_ready",
    "blocked",
]

# These thresholds are preliminary product defaults.
# They must not be confused with validated model performance.
READY_THRESHOLD = 80.0
ATTENTION_THRESHOLD = 50.0


class ValidationEvidence(BaseModel):
    """Evidence that a task's validation checks were completed."""

    model_config = ConfigDict(extra="forbid")

    completed: bool = False
    strategy: str
    actual_split_checked: bool = False
    split_overlap_checked: bool = False
    preprocessing_boundary_checked: bool = False
    feature_availability_checked: bool = False
    temporal_order_checked: bool | None = None
    group_isolation_checked: bool | None = None
    unsupervised_protocol_checked: bool | None = None
    notes: list[str] = Field(default_factory=list)


class LeakageEvidence(BaseModel):
    """Evidence that the required leakage audit is complete."""

    model_config = ConfigDict(extra="forbid")

    completed: bool = False
    target_leakage_checked: bool = False
    split_leakage_checked: bool = False
    preprocessing_leakage_checked: bool = False
    prediction_time_availability_checked: bool = False
    notes: list[str] = Field(default_factory=list)


class FinalScoringRequest(BaseModel):
    """Validated inputs for the final scoring gate."""

    model_config = ConfigDict(extra="forbid")

    task: MLTask
    configuration: ReadinessConfiguration
    dimensions: list[ReadinessDimensionResult]
    issues: list[ReadinessIssue] = Field(
        default_factory=list
    )
    validation: ValidationEvidence
    leakage: LeakageEvidence

    # For clustering, the default target-suitability
    # dimension must be disabled and its weight explicitly
    # redistributed to applicable dimensions.
    clustering_weights_approved: bool = False

    @model_validator(mode="after")
    def validate_dimensions(self) -> "FinalScoringRequest":
        """Reject duplicate and unsupported dimensions."""

        names = [
            dimension.name
            for dimension in self.dimensions
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Duplicate dimension results are not allowed."
            )

        unknown = set(names) - KNOWN_DIMENSIONS

        if unknown:
            raise ValueError(
                "Unknown dimension results: "
                + ", ".join(sorted(unknown))
            )

        return self


def _required_validation_checks(
    task: MLTask,
    evidence: ValidationEvidence,
) -> dict[str, bool]:
    """Identify the validation evidence required by task."""

    checks = {
        "validation_completed": evidence.completed,
        "actual_split_checked": (
            evidence.actual_split_checked
        ),
        "split_overlap_checked": (
            evidence.split_overlap_checked
        ),
        "preprocessing_boundary_checked": (
            evidence.preprocessing_boundary_checked
        ),
        "feature_availability_checked": (
            evidence.feature_availability_checked
        ),
    }

    if task == "forecasting":
        checks["temporal_order_checked"] = (
            evidence.temporal_order_checked is True
        )

    if evidence.strategy == "group_split":
        checks["group_isolation_checked"] = (
            evidence.group_isolation_checked is True
        )

    if task == "clustering":
        checks["unsupervised_protocol_checked"] = (
            evidence.unsupervised_protocol_checked is True
        )

    return checks


def _required_leakage_checks(
    task: MLTask,
    evidence: LeakageEvidence,
) -> dict[str, bool]:
    """Identify mandatory leakage-audit evidence."""

    checks = {
        "leakage_audit_completed": evidence.completed,
        "split_leakage_checked": (
            evidence.split_leakage_checked
        ),
        "preprocessing_leakage_checked": (
            evidence.preprocessing_leakage_checked
        ),
    }

    if task != "clustering":
        checks["target_leakage_checked"] = (
            evidence.target_leakage_checked
        )

        checks["prediction_time_availability_checked"] = (
            evidence.prediction_time_availability_checked
        )

    return checks


def _score_status(
    score: float,
) -> FinalStatus:
    """Translate a complete score into a display status."""

    if score >= READY_THRESHOLD:
        return "ready"

    if score >= ATTENTION_THRESHOLD:
        return "needs_attention"

    return "not_ready"


def evaluate_final_scoring(
    request: FinalScoringRequest,
) -> dict[str, Any]:
    """Evaluate whether a final readiness score is publishable.

    If the gate is blocked, final_score remains None.
    A provisional score may be calculated from completed
    dimensions, but is never presented as the final score.
    """

    reasons: list[str] = []

    configuration = request.configuration

    # Revalidate the configuration in case its values
    # were changed after initial model construction.
    configuration = ReadinessConfiguration.model_validate(
        configuration.model_dump(mode="python")
    )

    configured_names = set(
        configuration.dimensions
    )

    unknown_configured = (
        configured_names - KNOWN_DIMENSIONS
    )

    if unknown_configured:
        reasons.append(
            "The configuration contains unknown dimensions."
        )

    enabled = {
        name: settings
        for name, settings
        in configuration.dimensions.items()
        if settings.enabled
    }

    enabled_weight = sum(
        settings.weight
        for settings in enabled.values()
    )

    if not math.isclose(
        enabled_weight,
        100.0,
        abs_tol=0.001,
    ):
        reasons.append(
            "Enabled dimension weights must total 100."
        )

    results = {
        dimension.name: dimension
        for dimension in request.dimensions
    }

    missing = sorted(
        set(enabled) - set(results)
    )

    if missing:
        reasons.append(
            "Missing enabled dimensions: "
            + ", ".join(missing)
        )

    # A disabled dimension must not contribute to the score.
    measured: list[dict[str, Any]] = []

    for name, settings in enabled.items():
        dimension = results.get(name)

        if dimension is None:
            continue

        if dimension.status == "not_applicable":
            reasons.append(
                f"Enabled dimension '{name}' is not applicable."
            )
            continue

        score = float(dimension.score)

        if not math.isfinite(score):
            reasons.append(
                f"Dimension '{name}' has a non-finite score."
            )
            continue

        if not 0.0 <= score <= 100.0:
            reasons.append(
                f"Dimension '{name}' is outside 0–100."
            )
            continue

        # Use the approved configuration weight rather
        # than trusting the weight in a result payload.
        weighted = score * settings.weight / 100.0

        measured.append({
            "name": name,
            "score": round(score, 4),
            "weight": settings.weight,
            "weighted_score": round(weighted, 4),
        })

    # Clustering has no target. Users must approve a
    # configuration that redistributes its default weight.
    if request.task == "clustering":
        target_settings = configuration.dimensions.get(
            "target_suitability"
        )

        if (
            target_settings is not None
            and target_settings.enabled
        ):
            reasons.append(
                "Clustering must disable target suitability "
                "and redistribute its weight."
            )

        if not request.clustering_weights_approved:
            reasons.append(
                "Clustering-specific weights require approval."
            )

    # A dimension cannot be finalized solely by supplying
    # a numeric score. Its corresponding audit must also
    # be complete.
    validation_checks = _required_validation_checks(
        request.task,
        request.validation,
    )

    leakage_checks = _required_leakage_checks(
        request.task,
        request.leakage,
    )

    for name, passed in validation_checks.items():
        if not passed:
            reasons.append(
                f"Required validation check incomplete: {name}."
            )

    for name, passed in leakage_checks.items():
        if not passed:
            reasons.append(
                f"Required leakage check incomplete: {name}."
            )

    # Treat every critical issue as blocking. High and
    # medium issues remain visible but do not automatically
    # prevent a completed score from being published.
    critical_issues = [
        issue
        for issue in request.issues
        if issue.severity == "critical"
    ]

    if critical_issues:
        reasons.append(
            f"{len(critical_issues)} critical issue(s) "
            "remain unresolved."
        )

    # Prevent a misleading score when the supplied results
    # are incomplete or cannot account for all weight.
    measured_weight = sum(
        item["weight"]
        for item in measured
    )

    if not math.isclose(
        measured_weight,
        100.0,
        abs_tol=0.001,
    ):
        reasons.append(
            "Measured dimensions do not cover 100% "
            "of the approved scoring weight."
        )

    provisional_score = (
        round(
            sum(
                item["weighted_score"]
                for item in measured
            )
            * 100.0
            / measured_weight,
            2,
        )
        if measured_weight > 0
        else None
    )

    # Preserve the order while removing duplicate reasons.
    reasons = list(
        dict.fromkeys(reasons)
    )

    publishable = len(reasons) == 0

    final_score = (
        round(
            sum(
                item["weighted_score"]
                for item in measured
            ),
            2,
        )
        if publishable
        else None
    )

    status: FinalStatus = (
        _score_status(final_score)
        if final_score is not None
        else "blocked"
    )

    return {
        "task": request.task,
        "publishable": publishable,
        "status": status,
        "final_score": final_score,
        "provisional_score": provisional_score,
        "measured_weight_percentage": round(
            measured_weight,
            4,
        ),
        "blocking_reasons": reasons,
        "critical_issue_count": len(
            critical_issues
        ),
        "dimensions": measured,
        "validation_checks": validation_checks,
        "leakage_checks": leakage_checks,
        "configuration": {
            "configuration_id": (
                configuration.configuration_id
            ),
            "name": configuration.name,
            "version": configuration.version,
        },
        "scoring_rules": {
            "ready_threshold": READY_THRESHOLD,
            "attention_threshold": ATTENTION_THRESHOLD,
            "weights_total": round(
                enabled_weight,
                4,
            ),
        },
    }
