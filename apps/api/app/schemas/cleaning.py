
"""NUMPA v2 — Cleaning Studio API schemas.

Defines validated request and response contracts for:
    - Cleaning pipeline previews
    - Approved cleaning execution
    - Individual operation audit reports
    - Before/after dataset summaries
    - Reusable fitted artifacts

These schemas describe API data only. The actual cleaning
logic lives in app.services.cleaning.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    OperationName,
    PipelineStep,
)


# Re-export these models so API endpoints can import
# their public contracts from one schema module.
__all__ = [
    "OperationName",
    "PipelineStep",
    "CleaningPipelineConfig",
    "CleaningPreviewRequest",
    "CleaningExecuteRequest",
    "DatasetSummary",
    "DatasetChanges",
    "CleaningStepReport",
    "CleaningPipelineReport",
    "CleaningResultResponse",
    "CleaningErrorResponse",
]


class CleaningPreviewRequest(BaseModel):
    """Request a non-persistent cleaning preview."""

    dataset_id: str = Field(
        min_length=1,
        description="Identifier of an uploaded dataset.",
    )

    pipeline: CleaningPipelineConfig

    preview_rows: int = Field(
        default=10,
        ge=0,
        le=100,
    )

    # Reuse previously fitted parameters when appropriate.
    fitted_artifacts: dict[
        str,
        dict[str, Any],
    ] = Field(default_factory=dict)


class CleaningExecuteRequest(BaseModel):
    """Request execution of an approved cleaning pipeline."""

    dataset_id: str = Field(
        min_length=1,
        description="Identifier of an uploaded dataset.",
    )

    pipeline: CleaningPipelineConfig

    preview_rows: int = Field(
        default=10,
        ge=0,
        le=100,
    )

    fitted_artifacts: dict[
        str,
        dict[str, Any],
    ] = Field(default_factory=dict)

    # The frontend must explicitly confirm execution.
    confirmed: bool = False

    @model_validator(mode="after")
    def require_confirmation(
        self,
    ) -> "CleaningExecuteRequest":
        """Reject execution without user confirmation."""

        if not self.confirmed:
            raise ValueError(
                "Cleaning execution requires explicit "
                "user confirmation."
            )

        return self


class DatasetSummary(BaseModel):
    """Lightweight summary of a dataset before or after cleaning."""

    rows: int = Field(ge=0)
    columns: int = Field(ge=0)

    column_names: list[str]

    missing_cells: int = Field(ge=0)

    missing_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    memory_usage_bytes: int = Field(
        ge=0
    )


class DatasetChanges(BaseModel):
    """Difference between the original and cleaned datasets."""

    # Negative values are possible when a future operation
    # adds rows or columns rather than removing them.
    rows_removed: int
    columns_removed: int

    # Negative means the cleaning reduced missing values.
    missing_cells_change: int


class CleaningStepReport(BaseModel):
    """Audit report for one pipeline step."""

    step_id: str
    position: int = Field(ge=1)
    operation: OperationName

    status: Literal[
        "completed",
        "skipped",
        "failed",
    ]

    # Completed steps contain their operation-specific report.
    report: dict[str, Any] | None = None

    # Disabled steps contain an explanation.
    reason: str | None = None

    # Failed steps contain the error message.
    error: str | None = None


class CleaningPipelineReport(BaseModel):
    """Combined audit report for an entire pipeline run."""

    pipeline_name: str

    status: Literal[
        "completed",
        "completed_with_errors",
    ]

    total_steps: int = Field(ge=0)
    completed_steps: int = Field(ge=0)
    skipped_steps: int = Field(ge=0)
    failed_steps: int = Field(ge=0)

    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)

    columns_before: int = Field(ge=0)
    columns_after: int = Field(ge=0)

    missing_before: int = Field(ge=0)
    missing_after: int = Field(ge=0)

    step_reports: list[CleaningStepReport]

    @model_validator(mode="after")
    def validate_step_totals(
        self,
    ) -> "CleaningPipelineReport":
        """Ensure the reported step counts are consistent."""

        counted = (
            self.completed_steps
            + self.skipped_steps
            + self.failed_steps
        )

        if counted != self.total_steps:
            raise ValueError(
                "Pipeline step counts do not match total_steps."
            )

        if len(self.step_reports) != self.total_steps:
            raise ValueError(
                "step_reports length does not match total_steps."
            )

        return self


class CleaningResultResponse(BaseModel):
    """API response for a preview or execution."""

    execution_id: str

    mode: Literal[
        "preview",
        "execute",
    ]

    status: Literal[
        "completed",
        "completed_with_errors",
    ]

    executed_at: datetime

    pipeline_name: str

    before: DatasetSummary
    after: DatasetSummary
    changes: DatasetChanges

    pipeline_report: CleaningPipelineReport

    # Fitted configurations for encoding, scaling and outliers.
    # Later versions will extend train/test-safe replay to
    # all stateful cleaning operations.
    fitted_artifacts: dict[
        str,
        dict[str, Any],
    ] = Field(default_factory=dict)

    # A limited sample of the resulting dataset.
    preview: list[
        dict[str, Any]
    ] = Field(default_factory=list)

    # Execution endpoints may populate this after saving
    # the cleaned dataset through the storage service.
    cleaned_dataset_id: str | None = None


class CleaningErrorResponse(BaseModel):
    """Consistent API error response."""

    detail: str

    # Optional failing step details.
    step_id: str | None = None
    operation: str | None = None
