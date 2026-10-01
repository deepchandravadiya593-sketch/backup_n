
"""NUMPA v2 — Cleaning Studio service engine.

Provides a stable interface between the API and the
ordered cleaning pipeline.

Responsibilities:
    - Validate incoming pipeline configurations.
    - Preview cleaning without modifying the original.
    - Execute approved cleaning pipelines.
    - Return before/after dataset summaries.
    - Preserve operation audit reports.
    - Return reusable fitted artifacts.

This module does not persist datasets or pipeline results.
Storage will be handled by the storage service and API.
"""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    PipelineExecutionError,
    execute_pipeline,
)


ExecutionMode = Literal["preview", "execute"]


class CleaningRequest(BaseModel):
    """Validated request for a Cleaning Studio operation."""

    pipeline: CleaningPipelineConfig

    # Preview returns a limited number of sample rows.
    preview_rows: int = Field(
        default=10,
        ge=0,
        le=100,
    )

    # Optional fitted parameters from a previous run.
    # For ML, fit on training data and reuse on other splits.
    fitted_artifacts: dict[
        str,
        dict[str, Any],
    ] = Field(default_factory=dict)


def _dataset_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a lightweight dataset summary."""

    rows, columns = dataframe.shape

    missing_cells = int(
        dataframe.isna().sum().sum()
    )

    total_cells = int(rows * columns)

    missing_percentage = (
        round(
            missing_cells / total_cells * 100,
            4,
        )
        if total_cells
        else 0.0
    )

    return {
        "rows": int(rows),
        "columns": int(columns),
        "column_names": [
            str(column)
            for column in dataframe.columns
        ],
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "memory_usage_bytes": int(
            dataframe.memory_usage(
                deep=True
            ).sum()
        ),
    }


def _safe_preview_value(
    value: Any,
) -> Any:
    """Convert common pandas values into JSON-friendly data."""

    # Missing values must be checked before scalar conversion.
    if value is None or value is pd.NA:
        return None

    if isinstance(
        value,
        (pd.Timestamp, pd.Timedelta),
    ):
        return str(value)

    # NumPy scalars expose item(), returning a Python scalar.
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    # pd.isna can return an array for nested objects.
    # Preview values are expected to be scalar.
    try:
        missing = pd.isna(value)

        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    # Fall back to a readable representation for uncommon
    # scalar types such as Decimal or date objects.
    return str(value)


def _build_preview(
    dataframe: pd.DataFrame,
    limit: int,
) -> list[dict[str, Any]]:
    """Build a JSON-friendly sample of the dataset."""

    records = dataframe.head(
        limit
    ).to_dict(
        orient="records"
    )

    return [
        {
            str(column): _safe_preview_value(value)
            for column, value in row.items()
        }
        for row in records
    ]


def _run_cleaning(
    dataframe: pd.DataFrame,
    request: CleaningRequest,
    *,
    mode: ExecutionMode,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """Run the pipeline and prepare a structured service result."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Cleaning Studio requires a pandas DataFrame."
        )

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before cleaning."
        )

    # Keep the source dataset untouched.
    source = dataframe.copy(deep=True)

    before = _dataset_summary(
        source
    )

    # The pipeline already isolates individual steps and
    # prevents failed operations from mutating the source.
    cleaned, pipeline_report, fitted_artifacts = (
        execute_pipeline(
            source,
            request.pipeline,
            fitted_artifacts=deepcopy(
                request.fitted_artifacts
            ),
            preview=False,
        )
    )

    after = _dataset_summary(
        cleaned
    )

    # The execution ID identifies this particular run.
    # Persistent history will be added by the storage layer.
    execution_id = str(
        uuid4()
    )

    executed_at = datetime.now(
        timezone.utc
    ).isoformat()

    result: dict[str, Any] = {
        "execution_id": execution_id,
        "mode": mode,
        "status": pipeline_report["status"],
        "executed_at": executed_at,
        "pipeline_name": request.pipeline.name,
        "before": before,
        "after": after,
        "changes": {
            "rows_removed": (
                before["rows"] - after["rows"]
            ),
            "columns_removed": (
                before["columns"] - after["columns"]
            ),
            "missing_cells_change": (
                after["missing_cells"]
                - before["missing_cells"]
            ),
        },
        "pipeline_report": pipeline_report,
        "fitted_artifacts": fitted_artifacts,
        "preview": _build_preview(
            cleaned,
            request.preview_rows,
        ),
    }

    return cleaned, result


def preview_cleaning(
    dataframe: pd.DataFrame,
    request: CleaningRequest,
) -> dict[str, Any]:
    """Preview a pipeline without returning a committed dataset.

    The full pipeline runs on a copy. Only the structured
    report and a limited sample are returned.

    No dataset is saved or replaced by this function.
    """

    _, result = _run_cleaning(
        dataframe,
        request,
        mode="preview",
    )

    return result


def execute_cleaning(
    dataframe: pd.DataFrame,
    request: CleaningRequest,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """Execute an approved pipeline.

    Returns:
        cleaned:
            A new DataFrame ready for explicit storage.
        result:
            Structured summary, audit trail and artifacts.

    IMPORTANT:
        This function does not overwrite the original
        uploaded dataset or save the result automatically.
    """

    return _run_cleaning(
        dataframe,
        request,
        mode="execute",
    )


def validate_cleaning_request(
    data: dict[str, Any],
) -> CleaningRequest:
    """Validate a raw API request before executing cleaning."""

    return CleaningRequest.model_validate(
        data
    )
