
"""NUMPA v2 — Ordered data-cleaning pipeline.

Connects the eight individual cleaning operations.

Features:
    - User-controlled step order
    - Enable or disable individual steps
    - Transactional execution: failures never mutate the input
    - Individual and combined audit reports
    - Preview mode
    - Reusable fitted artifacts for supported operations

IMPORTANT:
For ML training, fit preprocessing steps on training data only.
"""

from copy import deepcopy
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field, model_validator, ConfigDict

from app.services.cleaning.operations.missing import (
    MissingValueConfig,
    clean_missing_values,
)
from app.services.cleaning.operations.duplicates import (
    DuplicateConfig,
    clean_duplicates,
)
from app.services.cleaning.operations.encoding import (
    EncodingConfig,
    encode_categories,
    transform_encoding,
)
from app.services.cleaning.operations.scaling import (
    ScalingConfig,
    scale_numeric,
    transform_scaler,
)
from app.services.cleaning.operations.outliers import (
    OutlierConfig,
    handle_outliers,
    transform_outliers,
)
from app.services.cleaning.operations.datatypes import (
    DataTypeConfig,
    convert_data_types,
)
from app.services.cleaning.operations.features import (
    FeatureSelectionConfig,
    select_features,
)
from app.services.cleaning.operations.text import (
    TextCleaningConfig,
    clean_text,
)


OperationName = Literal[
    "missing_values",
    "duplicates",
    "encoding",
    "scaling",
    "outliers",
    "datatypes",
    "features",
    "text",
]


# Each operation has its own validated configuration model.
CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "missing_values": MissingValueConfig,
    "duplicates": DuplicateConfig,
    "encoding": EncodingConfig,
    "scaling": ScalingConfig,
    "outliers": OutlierConfig,
    "datatypes": DataTypeConfig,
    "features": FeatureSelectionConfig,
    "text": TextCleaningConfig,
}


class PipelineStep(BaseModel):
    """One user-configured cleaning operation."""
    # Reject unknown fields such as obsolete step-level "keep" or "config".
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    operation: OperationName

    # Parameters are validated against the operation's
    # specific configuration model before execution.
    parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    enabled: bool = True

    @model_validator(mode="after")
    def validate_parameters(self) -> "PipelineStep":
        """Reject invalid operation configurations early."""

        model = CONFIG_MODELS[self.operation]

        # Validation does not modify the supplied parameters.
        model.model_validate(self.parameters)

        return self


class CleaningPipelineConfig(BaseModel):
    """An ordered collection of cleaning operations."""

    name: str = Field(
        default="Untitled Cleaning Pipeline",
        min_length=1,
        max_length=150,
    )

    steps: list[PipelineStep] = Field(
        default_factory=list
    )

    # Stop the pipeline on the first failed operation.
    # Partial results are never returned as successful output.
    stop_on_error: bool = True

    @model_validator(mode="after")
    def validate_step_ids(self) -> "CleaningPipelineConfig":
        """Ensure every step has a unique identifier."""

        ids = [
            step.step_id
            for step in self.steps
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Pipeline step IDs must be unique."
            )

        return self


class PipelineExecutionError(Exception):
    """An operation failed during pipeline execution."""

    def __init__(
        self,
        step_id: str,
        operation: str,
        message: str,
    ) -> None:
        self.step_id = step_id
        self.operation = operation
        self.message = message

        super().__init__(
            f"Step '{step_id}' ({operation}) failed: {message}"
        )


def _execute_step(
    dataframe: pd.DataFrame,
    step: PipelineStep,
    fitted_artifact: dict[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any] | None,
]:
    """Execute one operation and normalize its return values."""

    operation = step.operation
    params = step.parameters

    if operation == "missing_values":
        config = MissingValueConfig.model_validate(
            params
        )

        cleaned, report = clean_missing_values(
            dataframe,
            config,
        )

        return cleaned, report, None

    if operation == "duplicates":
        config = DuplicateConfig.model_validate(
            params
        )

        cleaned, report = clean_duplicates(
            dataframe,
            config,
        )

        return cleaned, report, None

    if operation == "encoding":
        if fitted_artifact is not None:
            cleaned, report = transform_encoding(
                dataframe,
                fitted_artifact,
            )

            return cleaned, report, fitted_artifact

        config = EncodingConfig.model_validate(
            params
        )

        return encode_categories(
            dataframe,
            config,
        )

    if operation == "scaling":
        if fitted_artifact is not None:
            cleaned, report = transform_scaler(
                dataframe,
                fitted_artifact,
            )

            return cleaned, report, fitted_artifact

        config = ScalingConfig.model_validate(
            params
        )

        return scale_numeric(
            dataframe,
            config,
        )

    if operation == "outliers":
        if fitted_artifact is not None:
            cleaned, report = transform_outliers(
                dataframe,
                fitted_artifact,
            )

            return cleaned, report, fitted_artifact

        config = OutlierConfig.model_validate(
            params
        )

        return handle_outliers(
            dataframe,
            config,
        )

    if operation == "datatypes":
        config = DataTypeConfig.model_validate(
            params
        )

        cleaned, report = convert_data_types(
            dataframe,
            config,
        )

        return cleaned, report, None

    if operation == "features":
        config = FeatureSelectionConfig.model_validate(
            params
        )

        cleaned, report = select_features(
            dataframe,
            config,
        )

        return cleaned, report, None

    if operation == "text":
        config = TextCleaningConfig.model_validate(
            params
        )

        cleaned, report = clean_text(
            dataframe,
            config,
        )

        return cleaned, report, None

    raise ValueError(
        f"Unsupported cleaning operation: {operation}"
    )


def execute_pipeline(
    dataframe: pd.DataFrame,
    config: CleaningPipelineConfig,
    *,
    fitted_artifacts: dict[str, dict[str, Any]] | None = None,
    preview: bool = False,
    preview_rows: int = 10,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    """Execute enabled pipeline steps in their configured order.

    Args:
        dataframe:
            Original dataset, which is never modified.
        config:
            Ordered pipeline configuration.
        fitted_artifacts:
            Optional previously fitted encoding, scaling and
            outlier configurations, indexed by step ID.
        preview:
            Include a small preview of the final dataset.
        preview_rows:
            Maximum number of preview rows.

    Returns:
        cleaned:
            Final cleaned DataFrame.
        report:
            Combined audit report and individual step reports.
        artifacts:
            Fitted parameters produced or reused by this run.

    Raises:
        PipelineExecutionError:
            If a step fails and stop_on_error is enabled.

    Note:
        Other operations currently run directly on the
        supplied dataset. Full train/test-safe replay for
        every operation will be added to the pipeline engine.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not isinstance(config, CleaningPipelineConfig):
        raise TypeError(
            "Expected a CleaningPipelineConfig instance."
        )

    if preview_rows < 0 or preview_rows > 100:
        raise ValueError(
            "preview_rows must be between 0 and 100."
        )

    # Protect the uploaded dataset from all operations.
    working = dataframe.copy(deep=True)

    # Deep-copy incoming artifacts to protect the caller.
    artifacts = deepcopy(
        fitted_artifacts
        if fitted_artifacts is not None
        else {}
    )

    step_reports: list[dict[str, Any]] = []
    failed_steps: list[dict[str, Any]] = []

    rows_before = int(len(dataframe))
    columns_before = int(len(dataframe.columns))
    missing_before = int(
        dataframe.isna().sum().sum()
    )

    for position, step in enumerate(
        config.steps,
        start=1,
    ):
        if not step.enabled:
            step_reports.append({
                "step_id": step.step_id,
                "position": position,
                "operation": step.operation,
                "status": "skipped",
                "reason": "Step disabled by user.",
            })

            continue

        # A failed operation must not change the working
        # dataset or overwrite existing fitted artifacts.
        step_input = working.copy(deep=True)

        try:
            cleaned, operation_report, artifact = (
                _execute_step(
                    step_input,
                    step,
                    artifacts.get(step.step_id),
                )
            )

            if not isinstance(cleaned, pd.DataFrame):
                raise TypeError(
                    "Cleaning operation did not return a DataFrame."
                )

            # Commit only after the operation succeeds.
            working = cleaned

            if artifact is not None:
                artifacts[step.step_id] = deepcopy(
                    artifact
                )

            step_reports.append({
                "step_id": step.step_id,
                "position": position,
                "operation": step.operation,
                "status": "completed",
                "report": operation_report,
            })

        except (
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
        ) as exc:
            failure = {
                "step_id": step.step_id,
                "position": position,
                "operation": step.operation,
                "status": "failed",
                "error": str(exc),
            }

            failed_steps.append(failure)
            step_reports.append(failure)

            if config.stop_on_error:
                raise PipelineExecutionError(
                    step_id=step.step_id,
                    operation=step.operation,
                    message=str(exc),
                ) from exc

            # Continue with the last successful dataset.
            continue

    completed_count = sum(
        item["status"] == "completed"
        for item in step_reports
    )

    skipped_count = sum(
        item["status"] == "skipped"
        for item in step_reports
    )

    rows_after = int(len(working))
    columns_after = int(len(working.columns))
    missing_after = int(
        working.isna().sum().sum()
    )

    report: dict[str, Any] = {
        "pipeline_name": config.name,
        "status": (
            "completed_with_errors"
            if failed_steps
            else "completed"
        ),
        "total_steps": len(config.steps),
        "completed_steps": completed_count,
        "skipped_steps": skipped_count,
        "failed_steps": len(failed_steps),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "columns_before": columns_before,
        "columns_after": columns_after,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "step_reports": step_reports,
    }

    if preview:
        # Convert preview rows into ordinary dictionaries.
        # JSON-safe conversion will be handled by the API
        # response layer.
        report["preview"] = working.head(
            preview_rows
        ).to_dict(
            orient="records"
        )

    return working, report, artifacts
