
"""NUMPA v2 - ML Readiness API schemas.

Defines request and response contracts for classification,
regression, time-series forecasting and clustering.

Scoring is deterministic. These schemas describe assessments;
the actual scoring engine will be implemented separately.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Supported ML tasks and validation strategies.
MLTask = Literal[
    "classification",
    "regression",
    "forecasting",
    "clustering",
]

ValidationStrategy = Literal[
    "auto",
    "random_split",
    "stratified_split",
    "time_series_split",
    "group_split",
    "cross_validation",
    "none",
]

ReadinessStatus = Literal[
    "ready",
    "needs_attention",
    "not_ready",
    "not_applicable",
]

IssueSeverity = Literal[
    "critical",
    "high",
    "medium",
    "low",
    "info",
]


# The default dimensions and their initial weights.
# Task-specific configurations may adjust these weights.
DEFAULT_DIMENSION_WEIGHTS = {
    "completeness": 20.0,
    "validity_consistency": 15.0,
    "record_integrity": 10.0,
    "feature_suitability": 15.0,
    "target_suitability": 15.0,
    "data_sufficiency": 10.0,
    "leakage_validation": 15.0,
}


class ScoringDimensionConfig(BaseModel):
    """User-configurable rules for one readiness dimension."""

    model_config = ConfigDict(extra="forbid")

    weight: float = Field(
        ge=0,
        le=100,
        description="Contribution to the overall readiness score.",
    )

    enabled: bool = True

    # Threshold values are stored as configuration data.
    # Their meaning is defined by the scoring engine.
    thresholds: dict[str, float] = Field(
        default_factory=dict,
    )


class ReadinessConfiguration(BaseModel):
    """A versioned, reusable readiness-scoring configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration_id: str | None = None
    name: str = Field(
        default="NUMPA Default",
        min_length=1,
        max_length=120,
    )
    version: int = Field(default=1, ge=1)

    dimensions: dict[str, ScoringDimensionConfig] = Field(
        default_factory=lambda: {
            name: ScoringDimensionConfig(weight=weight)
            for name, weight in DEFAULT_DIMENSION_WEIGHTS.items()
        }
    )

    # Additional task- or algorithm-specific parameters.
    # The engine must explicitly recognize a parameter before
    # applying it to a scoring calculation.
    parameters: dict[str, Any] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_weights(self) -> ReadinessConfiguration:
        """Require enabled dimension weights to total 100."""

        if not self.dimensions:
            raise ValueError(
                "At least one scoring dimension is required."
            )

        total = sum(
            dimension.weight
            for dimension in self.dimensions.values()
            if dimension.enabled
        )

        if abs(total - 100.0) > 0.001:
            raise ValueError(
                "Enabled scoring dimension weights must total 100. "
                f"Current total: {total:.3f}"
            )

        return self


class ReadinessAssessmentRequest(BaseModel):
    """Request an initial or final ML readiness assessment."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    task: MLTask

    # Clustering does not require a target column.
    target_column: str | None = None

    # Optional algorithm awareness, e.g. random_forest or kmeans.
    intended_algorithm: str | None = None

    # Forecasting usually requires a chronological column.
    datetime_column: str | None = None

    # Group-aware validation prevents leakage across groups.
    group_column: str | None = None

    # Columns to exclude from model input features.
    excluded_columns: list[str] = Field(
        default_factory=list,
    )

    validation_strategy: ValidationStrategy = "auto"

    # An omitted configuration means use NUMPA's built-in
    # task-specific defaults when the engine is implemented.
    configuration: ReadinessConfiguration | None = None

    @model_validator(mode="after")
    def validate_task_requirements(
        self,
    ) -> ReadinessAssessmentRequest:
        """Validate essential task-specific request fields."""

        if self.task in (
            "classification",
            "regression",
            "forecasting",
        ):
            if not self.target_column:
                raise ValueError(
                    f"A target column is required for {self.task}."
                )

        if self.task == "clustering" and self.target_column:
            raise ValueError(
                "Clustering does not use a target column."
            )

        if self.task == "forecasting" and not self.datetime_column:
            raise ValueError(
                "Forecasting requires a datetime column."
            )

        if (
            self.validation_strategy == "time_series_split"
            and not self.datetime_column
        ):
            raise ValueError(
                "Time-series validation requires a datetime column."
            )

        if (
            self.validation_strategy == "group_split"
            and not self.group_column
        ):
            raise ValueError(
                "Group validation requires a group column."
            )

        # Prevent duplicate exclusions and obvious role conflicts.
        if len(self.excluded_columns) != len(
            set(self.excluded_columns)
        ):
            raise ValueError(
                "Excluded columns must not contain duplicates."
            )

        if (
            self.target_column
            and self.target_column in self.excluded_columns
        ):
            raise ValueError(
                "The target column cannot also be excluded."
            )

        return self


class ReadinessIssue(BaseModel):
    """An explainable issue discovered during assessment."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: IssueSeverity
    dimension: str

    message: str

    # The affected column is optional for dataset-wide issues.
    column: str | None = None

    recommendation: str | None = None

    # Machine-readable evidence for reports and auditing.
    evidence: dict[str, Any] = Field(
        default_factory=dict,
    )


class ReadinessDimensionResult(BaseModel):
    """Scoring result for one readiness dimension."""

    model_config = ConfigDict(extra="forbid")

    name: str

    score: float = Field(
        ge=0,
        le=100,
    )

    weight: float = Field(
        ge=0,
        le=100,
    )

    weighted_score: float = Field(
        ge=0,
        le=100,
    )

    status: ReadinessStatus

    # Explain how this dimension received its score.
    explanation: str

    issues: list[ReadinessIssue] = Field(
        default_factory=list,
    )

    metrics: dict[str, Any] = Field(
        default_factory=dict,
    )


class ReadinessAssessmentResponse(BaseModel):
    """Complete, explainable ML readiness assessment."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    dataset_id: str

    task: MLTask
    target_column: str | None = None
    intended_algorithm: str | None = None
    validation_strategy: ValidationStrategy

    configuration_id: str | None = None
    configuration_name: str
    configuration_version: int = Field(ge=1)

    overall_score: float = Field(
        ge=0,
        le=100,
    )

    status: ReadinessStatus

    dimensions: list[ReadinessDimensionResult]

    issues: list[ReadinessIssue] = Field(
        default_factory=list,
    )

    # Important warnings may block an otherwise high score.
    blocking_issues: list[ReadinessIssue] = Field(
        default_factory=list,
    )

    recommendations: list[str] = Field(
        default_factory=list,
    )

    assessed_at: datetime

    # Records the rules and input context used for auditability.
    audit: dict[str, Any] = Field(
        default_factory=dict,
    )


class ReadinessComparisonRequest(BaseModel):
    """Compare assessments before and after cleaning."""

    model_config = ConfigDict(extra="forbid")

    before_assessment_id: str = Field(min_length=1)
    after_assessment_id: str = Field(min_length=1)


class ReadinessDimensionChange(BaseModel):
    """Describe the change in one readiness dimension."""

    name: str

    before_score: float = Field(ge=0, le=100)
    after_score: float = Field(ge=0, le=100)

    # Negative changes are allowed; cleaning can make data worse.
    change: float = Field(ge=-100, le=100)


class ReadinessComparisonResponse(BaseModel):
    """Before-and-after assessment comparison."""

    before_assessment_id: str
    after_assessment_id: str

    before_dataset_id: str
    after_dataset_id: str

    task: MLTask

    before_score: float = Field(ge=0, le=100)
    after_score: float = Field(ge=0, le=100)

    score_change: float = Field(ge=-100, le=100)

    dimension_changes: list[ReadinessDimensionChange]

    resolved_issues: list[ReadinessIssue] = Field(
        default_factory=list,
    )

    remaining_issues: list[ReadinessIssue] = Field(
        default_factory=list,
    )

    new_issues: list[ReadinessIssue] = Field(
        default_factory=list,
    )

    compared_at: datetime


class ReadinessErrorResponse(BaseModel):
    """Consistent error response for readiness endpoints."""

    detail: str
