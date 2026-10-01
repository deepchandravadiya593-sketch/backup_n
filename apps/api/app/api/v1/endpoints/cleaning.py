
"""NUMPA v2 - Cleaning Studio API endpoints.

Supports:
- Previewing cleaning pipelines without saving changes.
- Executing approved pipelines and saving cleaned datasets.
- Automatically saving successful execution history.
- Listing and retrieving previous executions.
- Generating PDFs from submitted or saved reports.

Execution history currently uses temporary in-memory storage.
"""

from __future__ import annotations

from io import BytesIO
import re
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.api.v1.endpoints.datasets import get_dataset_dataframe

from app.schemas.cleaning import (
    CleaningExecuteRequest,
    CleaningPreviewRequest,
    CleaningResultResponse,
)

from app.services.cleaning.engine import (
    CleaningRequest,
    execute_cleaning,
    preview_cleaning,
)

from app.services.cleaning.pipeline import PipelineExecutionError

from app.services.reports.cleaning_pdf import (
    generate_cleaning_report_pdf,
)

from app.services.storage.datasets import (
    delete_dataset as storage_delete_dataset,
    save_dataset,
)

from app.services.storage.executions import (
    ExecutionAlreadyExistsError,
    ExecutionNotFoundError,
    ExecutionStorageLimitError,
    get_execution,
    list_executions,
    save_execution,
)


# ---------------------------------------------------------
# API router
# ---------------------------------------------------------

router = APIRouter(
    prefix="/cleaning",
    tags=["Cleaning Studio"],
)


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------

def _build_engine_request(
    request: CleaningPreviewRequest | CleaningExecuteRequest,
) -> CleaningRequest:
    """Convert an API request to the cleaning engine format."""

    return CleaningRequest(
        pipeline=request.pipeline,
        preview_rows=request.preview_rows,
        fitted_artifacts=request.fitted_artifacts,
    )


def _handle_cleaning_error(exc: Exception) -> None:
    """Convert expected cleaning errors into HTTP responses."""

    if isinstance(exc, PipelineExecutionError):
        raise HTTPException(
            status_code=422,
            detail={
                "message": exc.message,
                "step_id": exc.step_id,
                "operation": exc.operation,
            },
        ) from exc

    if isinstance(
        exc,
        (ValueError, TypeError, ValidationError),
    ):
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Unexpected errors should reach FastAPI's error handler.
    raise exc


def _cleaning_pdf_filename(
    execution_id: str,
) -> str:
    """Create a safe PDF filename from an execution ID."""

    safe_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        execution_id,
    ).strip("_")

    safe_id = safe_id[:80] or "execution"

    return f"numpa_cleaning_{safe_id}.pdf"


def _pdf_response(
    pdf_bytes: bytes,
    execution_id: str,
) -> StreamingResponse:
    """Return generated PDF bytes as a download."""

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=500,
            detail="Cleaning PDF generation failed.",
        )

    filename = _cleaning_pdf_filename(
        execution_id
    )

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Content-Length": str(len(pdf_bytes)),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _load_execution_or_404(
    execution_id: str,
) -> dict[str, Any]:
    """Retrieve a saved execution with API error handling."""

    try:
        return get_execution(execution_id)

    except ExecutionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------
# 1. Preview cleaning
# ---------------------------------------------------------

@router.post(
    "/preview",
    response_model=CleaningResultResponse,
    summary="Preview a cleaning pipeline",
)
def preview_dataset_cleaning(
    request: CleaningPreviewRequest,
) -> CleaningResultResponse:
    """Preview cleaning without changing stored datasets."""

    # Retrieve an independent copy of the source dataset.
    dataframe = get_dataset_dataframe(
        request.dataset_id
    )

    engine_request = _build_engine_request(
        request
    )

    try:
        result = preview_cleaning(
            dataframe,
            engine_request,
        )

        # Preview mode never creates a cleaned dataset
        # or a saved execution-history record.
        result["cleaned_dataset_id"] = None

        return CleaningResultResponse.model_validate(
            result
        )

    except Exception as exc:
        _handle_cleaning_error(exc)
        raise


# ---------------------------------------------------------
# 2. Execute cleaning and save history
# ---------------------------------------------------------

@router.post(
    "/execute",
    response_model=CleaningResultResponse,
    summary="Execute and save a cleaning pipeline",
)
def execute_dataset_cleaning(
    request: CleaningExecuteRequest,
) -> CleaningResultResponse:
    """Execute an approved pipeline and save its results."""

    # CleaningExecuteRequest already requires confirmation.
    # Check explicitly as an additional safety measure.
    if not request.confirmed:
        raise HTTPException(
            status_code=422,
            detail=(
                "Confirm the cleaning operation "
                "before execution."
            ),
        )

    dataframe = get_dataset_dataframe(
        request.dataset_id
    )

    engine_request = _build_engine_request(
        request
    )

    try:
        cleaned_dataframe, result = execute_cleaning(
            dataframe,
            engine_request,
        )

    except Exception as exc:
        _handle_cleaning_error(exc)
        raise

    # Do not store output from an incomplete pipeline.
    if result.get("status") != "completed":
        raise HTTPException(
            status_code=422,
            detail=(
                "The cleaning pipeline did not complete "
                "successfully. No cleaned dataset was saved."
            ),
        )

    # Give the cleaned dataset a recognizable name.
    cleaned_name = (
        f"cleaned_{request.dataset_id[:8]}_"
        f"{result['execution_id'][:8]}.csv"
    )

    try:
        # Save a separate dataset with parent lineage.
        metadata = save_dataset(
            cleaned_dataframe,
            name=cleaned_name,
            kind="cleaned",
            parent_dataset_id=request.dataset_id,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "The original dataset is "
                "no longer available."
            ),
        ) from exc

    except ValueError as exc:
        message = str(exc)

        if "storage is full" in message.lower():
            raise HTTPException(
                status_code=507,
                detail=message,
            ) from exc

        raise HTTPException(
            status_code=422,
            detail=message,
        ) from exc

    cleaned_dataset_id = metadata["dataset_id"]

    # Attach the saved dataset ID before validating
    # or storing the execution report.
    result["cleaned_dataset_id"] = cleaned_dataset_id

    # Record the original dataset ID explicitly.
    # This supports execution-history filtering.
    result["dataset_id"] = request.dataset_id

    try:
        # Validate the final API response first.
        response = CleaningResultResponse.model_validate(
            result
        )

        # Convert the validated model to JSON-safe data.
        # Include the original dataset ID for history.
        execution_record = response.model_dump(
            mode="json"
        )

        execution_record["dataset_id"] = (
            request.dataset_id
        )

        # Store the execution only after the cleaned
        # dataset exists and response validation succeeds.
        save_execution(execution_record)

    except ExecutionAlreadyExistsError as exc:
        # Avoid leaving a cleaned dataset without
        # a corresponding execution-history record.
        storage_delete_dataset(cleaned_dataset_id)

        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ExecutionStorageLimitError as exc:
        storage_delete_dataset(cleaned_dataset_id)

        raise HTTPException(
            status_code=507,
            detail=str(exc),
        ) from exc

    except Exception:
        # Roll back the newly created dataset if response
        # validation or history storage unexpectedly fails.
        storage_delete_dataset(cleaned_dataset_id)
        raise

    return response


# ---------------------------------------------------------
# 3. Generate PDF from a submitted execution report
# ---------------------------------------------------------

@router.post(
    "/report/pdf",
    summary="Generate a cleaning report PDF",
    description=(
        "Generate a PDF from a submitted execution result. "
        "The submitted report is not independently verified."
    ),
)
def download_cleaning_report_pdf(
    execution_result: dict[str, Any] = Body(...),
) -> StreamingResponse:
    """Convert a submitted cleaning report into a PDF."""

    execution_id = execution_result.get(
        "execution_id"
    )

    if (
        not isinstance(execution_id, str)
        or not execution_id.strip()
    ):
        raise HTTPException(
            status_code=422,
            detail="A valid execution_id is required.",
        )

    try:
        pdf_bytes = generate_cleaning_report_pdf(
            execution_result
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return _pdf_response(
        pdf_bytes,
        execution_id,
    )


# ---------------------------------------------------------
# 4. List saved execution history
# ---------------------------------------------------------

@router.get(
    "/history",
    summary="List cleaning execution history",
)
def get_cleaning_execution_history(
    dataset_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List previous cleaning runs, newest first."""

    return list_executions(
        dataset_id=dataset_id,
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------
# 5. Retrieve one saved execution
# ---------------------------------------------------------

@router.get(
    "/history/{execution_id}",
    summary="Retrieve a saved cleaning execution",
)
def get_cleaning_execution_report(
    execution_id: str,
) -> dict[str, Any]:
    """Retrieve a previous execution by its ID."""

    return _load_execution_or_404(
        execution_id
    )


# ---------------------------------------------------------
# 6. Download a PDF from saved execution history
# ---------------------------------------------------------

@router.get(
    "/history/{execution_id}/report/pdf",
    summary="Download a saved cleaning execution PDF",
)
def download_saved_cleaning_report_pdf(
    execution_id: str,
) -> StreamingResponse:
    """Generate a PDF using a server-stored execution."""

    # Unlike the POST PDF endpoint, this endpoint
    # reads the report from NUMPA's history storage.
    execution = _load_execution_or_404(
        execution_id
    )

    try:
        pdf_bytes = generate_cleaning_report_pdf(
            execution
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return _pdf_response(
        pdf_bytes,
        execution_id,
    )
