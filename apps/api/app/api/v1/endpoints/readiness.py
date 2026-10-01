
"""NUMPA v2 - ML Readiness API endpoints.

Exposes provisional assessments and versioned scoring profiles.

Important:
- The assessment engine does not yet produce final readiness scores.
- Configuration history is immutable.
- Assessment results and configurations currently live in memory.
- Authentication and persistent storage will be added later.
"""

from __future__ import annotations

from threading import RLock
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.readiness import (
    ReadinessAssessmentRequest,
    ReadinessConfiguration,
)
from app.services.readiness.configurations import (
    ConfigurationLimitError,
    ConfigurationNotFoundError,
    configuration_manager,
)
from app.services.readiness.engine import assess_readiness
from app.services.storage.datasets import get_dataset


router = APIRouter(
    prefix="/readiness",
    tags=["ML Readiness"],
)


# Temporary in-memory assessment history.
# Persistent, user-scoped storage will replace this later.
_assessments: dict[str, dict[str, Any]] = {}
_assessment_lock = RLock()
MAX_SAVED_ASSESSMENTS = 200


class AssessmentCreateRequest(ReadinessAssessmentRequest):
    """Assessment request with optional actual split positions."""

    train_positions: list[int] | None = None
    validation_positions: list[int] | None = None


class ConfigurationCreateRequest(BaseModel):
    """Create a new scoring configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration: ReadinessConfiguration
    created_by: str | None = None


class ConfigurationVersionRequest(BaseModel):
    """Save a new immutable configuration version."""

    model_config = ConfigDict(extra="forbid")

    configuration: ReadinessConfiguration
    expected_version: int | None = Field(
        default=None,
        ge=1,
    )


class ConfigurationDuplicateRequest(BaseModel):
    """Duplicate a previously saved configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=120,
    )
    version: int | None = Field(
        default=None,
        ge=1,
    )
    created_by: str | None = None


def _get_dataset_or_404(dataset_id: str):
    """Retrieve a stored dataset or raise an HTTP 404."""

    try:
        dataframe = get_dataset(dataset_id)
    except (KeyError, LookupError):
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        ) from None

    if dataframe is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return dataframe


def _get_configuration_or_404(
    configuration_id: str,
    *,
    version: int | None = None,
) -> ReadinessConfiguration:
    """Retrieve a scoring configuration with HTTP error mapping."""

    try:
        return configuration_manager.get(
            configuration_id,
            version=version,
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


def _get_assessment_or_404(
    assessment_id: str,
) -> dict[str, Any]:
    """Return an independent copy of a stored assessment."""

    from copy import deepcopy

    with _assessment_lock:
        assessment = _assessments.get(
            assessment_id
        )

        if assessment is None:
            raise HTTPException(
                status_code=404,
                detail="Assessment not found.",
            )

        return deepcopy(assessment)


@router.post(
    "/assess",
    summary="Create a provisional ML readiness assessment",
)
def create_assessment(
    request: AssessmentCreateRequest,
) -> dict[str, Any]:
    """Assess a stored dataset using the selected ML task.

    A saved configuration may be referenced by ID and version.
    Alternatively, the caller may supply a complete inline
    configuration through the existing request schema.
    """

    dataframe = _get_dataset_or_404(
        request.dataset_id
    )

    # Copy the request into the engine's original schema.
    # Split positions are separate optional engine arguments.
    engine_request = ReadinessAssessmentRequest.model_validate(
        request.model_dump(
            exclude={
                "train_positions",
                "validation_positions",
            }
        )
    )

    # When a configuration ID is supplied, resolve the
    # immutable saved version instead of trusting an inline
    # payload that could differ from stored history.
    supplied = engine_request.configuration

    if (
        supplied is not None
        and supplied.configuration_id is not None
    ):
        saved = _get_configuration_or_404(
            supplied.configuration_id,
            version=supplied.version,
        )

        if (
            supplied.model_dump(
                exclude={
                    "configuration_id",
                    "version",
                }
            )
            != saved.model_dump(
                exclude={
                    "configuration_id",
                    "version",
                }
            )
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "The supplied configuration differs from "
                    "the saved configuration version."
                ),
            )

        engine_request.configuration = saved

    try:
        result = assess_readiness(
            dataframe,
            engine_request,
            train_positions=request.train_positions,
            validation_positions=request.validation_positions,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Save the report for later retrieval and comparison.
    with _assessment_lock:
        if len(_assessments) >= MAX_SAVED_ASSESSMENTS:
            raise HTTPException(
                status_code=507,
                detail=(
                    "The temporary assessment storage "
                    "limit has been reached."
                ),
            )

        from copy import deepcopy

        _assessments[result["assessment_id"]] = (
            deepcopy(result)
        )

    return result


@router.get(
    "/assessments",
    summary="List saved provisional assessments",
)
def list_assessments(
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """List saved assessments, optionally filtered by dataset."""

    with _assessment_lock:
        items = [
            {
                "assessment_id": item["assessment_id"],
                "dataset_id": item["dataset_id"],
                "task": item["task"],
                "assessment_status": item[
                    "assessment_status"
                ],
                "provisional_score": item[
                    "provisional_score"
                ],
                "assessed_at": item["assessed_at"],
            }
            for item in _assessments.values()
            if (
                dataset_id is None
                or item["dataset_id"] == dataset_id
            )
        ]

    items.sort(
        key=lambda item: item["assessed_at"],
        reverse=True,
    )

    return {
        "count": len(items),
        "assessments": items,
    }


@router.get(
    "/assessments/{assessment_id}",
    summary="Retrieve a saved assessment",
)
def get_assessment(
    assessment_id: str,
) -> dict[str, Any]:
    """Return the full saved provisional assessment."""

    return _get_assessment_or_404(
        assessment_id
    )


@router.get(
    "/configurations/default",
    summary="Get NUMPA's default scoring configuration",
)
def get_default_configuration() -> dict[str, Any]:
    """Return the built-in default without saving it."""

    return configuration_manager.get_default().model_dump(
        mode="json"
    )


@router.post(
    "/configurations",
    status_code=201,
    summary="Create a scoring configuration",
)
def create_configuration(
    request: ConfigurationCreateRequest,
) -> dict[str, Any]:
    """Save a new configuration as immutable version 1."""

    try:
        return configuration_manager.create(
            request.configuration,
            created_by=request.created_by,
        )
    except ConfigurationLimitError as exc:
        raise HTTPException(
            status_code=507,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@router.get(
    "/configurations",
    summary="List scoring configurations",
)
def list_configurations(
    include_archived: bool = False,
) -> dict[str, Any]:
    """List configuration metadata."""

    items = configuration_manager.list_configurations(
        include_archived=include_archived,
    )

    return {
        "count": len(items),
        "configurations": items,
    }


@router.get(
    "/configurations/{configuration_id}/versions",
    summary="List configuration versions",
)
def list_configuration_versions(
    configuration_id: str,
) -> dict[str, Any]:
    """List all immutable versions, newest first."""

    try:
        versions = configuration_manager.list_versions(
            configuration_id
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return {
        "configuration_id": configuration_id,
        "count": len(versions),
        "versions": versions,
    }


@router.get(
    "/configurations/{configuration_id}",
    summary="Retrieve a scoring configuration",
)
def get_configuration(
    configuration_id: str,
    version: int | None = Query(
        default=None,
        ge=1,
    ),
) -> dict[str, Any]:
    """Retrieve the latest or a specific configuration version."""

    configuration = _get_configuration_or_404(
        configuration_id,
        version=version,
    )

    return configuration.model_dump(
        mode="json"
    )


@router.post(
    "/configurations/{configuration_id}/versions",
    status_code=201,
    summary="Create a new configuration version",
)
def create_configuration_version(
    configuration_id: str,
    request: ConfigurationVersionRequest,
) -> dict[str, Any]:
    """Save an approved update without changing old versions."""

    try:
        return configuration_manager.create_version(
            configuration_id,
            request.configuration,
            expected_version=request.expected_version,
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ConfigurationLimitError as exc:
        raise HTTPException(
            status_code=507,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        status = (
            409
            if "conflict" in str(exc).lower()
            else 422
        )

        raise HTTPException(
            status_code=status,
            detail=str(exc),
        ) from exc


@router.post(
    "/configurations/{configuration_id}/duplicate",
    status_code=201,
    summary="Duplicate a saved configuration",
)
def duplicate_configuration(
    configuration_id: str,
    request: ConfigurationDuplicateRequest,
) -> dict[str, Any]:
    """Copy a saved version into a new configuration."""

    try:
        return configuration_manager.duplicate(
            configuration_id,
            name=request.name,
            version=request.version,
            created_by=request.created_by,
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ConfigurationLimitError as exc:
        raise HTTPException(
            status_code=507,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@router.post(
    "/configurations/{configuration_id}/archive",
    summary="Archive a scoring configuration",
)
def archive_configuration(
    configuration_id: str,
) -> dict[str, Any]:
    """Archive a configuration without deleting its history."""

    try:
        return configuration_manager.archive(
            configuration_id
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.post(
    "/configurations/{configuration_id}/restore",
    summary="Restore an archived scoring configuration",
)
def restore_configuration(
    configuration_id: str,
) -> dict[str, Any]:
    """Restore an archived configuration."""

    try:
        return configuration_manager.restore(
            configuration_id
        )
    except ConfigurationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

# ---------------------------------------------------------
# Readiness assessment comparison
# ---------------------------------------------------------

from app.schemas.readiness import ReadinessComparisonRequest
from app.services.reports.comparison import compare_assessments


@router.post(
    "/compare",
    summary="Compare two ML readiness assessments",
    description=(
        "Compare an initial assessment with a later assessment. "
        "Score changes are reported only when the two assessments "
        "use compatible tasks, configurations and measurement coverage."
    ),
)
def compare_readiness_assessments(
    request: ReadinessComparisonRequest,
) -> dict[str, Any]:
    """Generate a before-and-after assessment comparison.

    Both assessments must already exist in NUMPA's temporary
    assessment store. The comparison service decides whether
    their scores are directly comparable.
    """

    # Prevent users from comparing an assessment with itself.
    if (
        request.before_assessment_id
        == request.after_assessment_id
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Select two different assessments "
                "for comparison."
            ),
        )

    # Reuse the existing lookup helper so missing
    # assessments produce consistent HTTP 404 responses.
    before = _get_assessment_or_404(
        request.before_assessment_id
    )

    after = _get_assessment_or_404(
        request.after_assessment_id
    )

    try:
        comparison = compare_assessments(
            before,
            after,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return comparison

# ---------------------------------------------------------
# File 47 - ML Readiness PDF Download
# ---------------------------------------------------------

from copy import deepcopy
from io import BytesIO

from fastapi.responses import StreamingResponse

from app.services.reports.readiness_pdf import (
    generate_readiness_assessment_pdf,
)


@router.get(
    "/assessments/{assessment_id}/report/pdf",
    summary="Download ML readiness assessment PDF",
    description=(
        "Download a PDF containing the saved assessment's "
        "score, dimension results, detected issues, "
        "recommendations and outstanding checks."
    ),
)
def download_readiness_assessment_pdf(
    assessment_id: str,
) -> StreamingResponse:
    """Generate a PDF for a previously saved assessment."""

    # Retrieve a defensive copy under the existing
    # assessment-store lock.
    with _assessment_lock:
        stored_assessment = _assessments.get(
            assessment_id
        )

        assessment = (
            deepcopy(stored_assessment)
            if stored_assessment is not None
            else None
        )

    # Return a clear error for unknown assessment IDs.
    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail="Readiness assessment not found.",
        )

    # Generate the PDF using File 46's report service.
    # The generator preserves the stored assessment
    # status and does not recalculate its score.
    try:
        pdf_bytes = generate_readiness_assessment_pdf(
            assessment
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Cannot generate this assessment's PDF: "
                f"{exc}"
            ),
        ) from exc

    # Check the generated file before returning it.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=500,
            detail="PDF generation failed.",
        )

    # Use the stored UUID as the download filename.
    # This avoids exposing user-controlled filenames
    # in HTTP response headers.
    filename = (
        f"numpa_readiness_{assessment_id}.pdf"
    )

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Content-Length": str(
                len(pdf_bytes)
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )

# ---------------------------------------------------------
# File 51 - ML Readiness Comparison PDF Download
# ---------------------------------------------------------

from copy import deepcopy
from io import BytesIO

from fastapi.responses import StreamingResponse

from app.schemas.readiness import (
    ReadinessComparisonRequest as ComparisonPDFRequest,
)
from app.services.reports.comparison_pdf import (
    generate_comparison_pdf_from_assessments,
)


@router.post(
    "/compare/report/pdf",
    summary="Download ML readiness comparison PDF",
    description=(
        "Compare two saved ML readiness assessments "
        "and download their before-and-after PDF report. "
        "Score changes are shown only when the assessments "
        "are directly comparable."
    ),
)
def download_readiness_comparison_pdf(
    request: ComparisonPDFRequest,
) -> StreamingResponse:
    """Generate a PDF comparing two saved assessments."""

    # Comparing an assessment with itself does not
    # represent a before-and-after workflow.
    if (
        request.before_assessment_id
        == request.after_assessment_id
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Select two different assessments "
                "for comparison."
            ),
        )

    # Retrieve both assessments under the existing lock.
    # Deep copies prevent the report generator from
    # accidentally modifying stored assessment records.
    with _assessment_lock:
        stored_before = _assessments.get(
            request.before_assessment_id
        )

        stored_after = _assessments.get(
            request.after_assessment_id
        )

        before = (
            deepcopy(stored_before)
            if stored_before is not None
            else None
        )

        after = (
            deepcopy(stored_after)
            if stored_after is not None
            else None
        )

    # Return 404 when either assessment is unavailable.
    if before is None:
        raise HTTPException(
            status_code=404,
            detail="Initial readiness assessment not found.",
        )

    if after is None:
        raise HTTPException(
            status_code=404,
            detail="Final readiness assessment not found.",
        )

    # File 50 delegates all comparison calculations
    # and compatibility decisions to File 42.
    try:
        pdf_bytes = generate_comparison_pdf_from_assessments(
            before,
            after,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Validate the generated file before returning it.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=500,
            detail="Comparison PDF generation failed.",
        )

    # Use a fixed filename to avoid placing untrusted
    # assessment IDs into HTTP response headers.
    filename = "numpa_readiness_comparison.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Content-Length": str(
                len(pdf_bytes)
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
